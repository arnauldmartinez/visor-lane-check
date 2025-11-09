import Foundation
import AVFoundation
import Speech
import Combine

final class SpeechWakeService: NSObject, ObservableObject {

    enum State: String { case idle, listening, recordingQuestion }

    @Published private(set) var state: State = .idle
    @Published private(set) var lastPartial: String = ""
    @Published private(set) var lastQuestion: String = ""

    var onWake: (() -> Void)?

    // Config
    private let wakeWord: String
    private let silenceWindow: TimeInterval

    // Speech
    private let audioEngine = AVAudioEngine()
    private var recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en_US"))
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?

    // State for slicing & silence detection
    private var silenceTimer: Timer?
    private var fullTranscriptLower: String = ""
    private var lastQuestionSlice: String = ""
    private var lastChangeAt = Date.distantPast
    private var wakeFiredForThisUtterance = false

    init(wakeWord: String = "visor", silenceWindow: TimeInterval = 0.35) {
        self.wakeWord = wakeWord.lowercased()
        self.silenceWindow = silenceWindow
    }

    // Permissions
    func requestPermissions(completion: @escaping (Bool) -> Void) {
        let askMic: (@escaping (Bool) -> Void) -> Void = { done in
            if #available(iOS 17.0, *) {
                AVAudioApplication.requestRecordPermission { granted in done(granted) }
            } else {
                AVAudioSession.sharedInstance().requestRecordPermission { granted in done(granted) }
            }
        }

        askMic { micGranted in
            guard micGranted else { DispatchQueue.main.async { completion(false) }; return }
            SFSpeechRecognizer.requestAuthorization { auth in
                DispatchQueue.main.async { completion(auth == .authorized) }
            }
        }
    }

    // Public control
    func start() {
        guard state == .idle else { return }
        requestPermissions { [weak self] ok in
            guard let self = self, ok else { return }
            self.beginSession()
        }
    }

    func stop() {
        invalidateSilenceTimer()
        request?.endAudio()
        if audioEngine.isRunning {
            audioEngine.stop()
            audioEngine.inputNode.removeTap(onBus: 0)
        }
        task?.cancel(); task = nil
        request = nil
        try? AVAudioSession.sharedInstance().setActive(false, options: [.notifyOthersOnDeactivation])
        DispatchQueue.main.async { self.state = .idle; self.lastPartial = "" }
    }

    // Internals
    private func beginSession() {
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.record, mode: .measurement, options: [.duckOthers])
            try session.setActive(true, options: [.notifyOthersOnDeactivation])
        } catch { print("SpeechWakeService audio session error:", error) }

        request = SFSpeechAudioBufferRecognitionRequest()
        request?.shouldReportPartialResults = true
        if #available(iOS 16.0, *) { request?.requiresOnDeviceRecognition = true }

        let input = audioEngine.inputNode
        let format = input.outputFormat(forBus: 0)
        input.removeTap(onBus: 0)
        input.installTap(onBus: 0, bufferSize: 2048, format: format) { [weak self] buffer, _ in
            guard let self = self, buffer.frameLength > 0 else { return } // zero-length guard
            self.request?.append(buffer)
        }

        do {
            audioEngine.prepare()
            try audioEngine.start()
        } catch { print("SpeechWakeService engine start error:", error) }

        state = .listening
        fullTranscriptLower = ""
        lastQuestionSlice = ""
        lastPartial = ""
        lastQuestion = ""
        wakeFiredForThisUtterance = false

        guard let request = request else { return }
        task = recognizer?.recognitionTask(with: request) { [weak self] result, error in
            guard let self = self else { return }

            if let error = error { print("SpeechWakeService recognizer error:", error.localizedDescription) }
            guard let result = result else { return }

            let text = result.bestTranscription.formattedString
            self.fullTranscriptLower = text.lowercased()
            DispatchQueue.main.async { self.lastPartial = text }

            switch self.state {
            case .listening:
                if let range = self.fullTranscriptLower.range(of: self.wakeWord, options: .backwards) {
                    if !self.wakeFiredForThisUtterance {
                        self.wakeFiredForThisUtterance = true
                        DispatchQueue.main.async { self.onWake?() }
                    }
                    let afterWake = text[range.upperBound...].trimmingCharacters(in: .whitespaces)
                    self.state = .recordingQuestion
                    self.lastQuestionSlice = afterWake
                    self.lastChangeAt = Date()
                    self.scheduleSilenceTimer()
                }

            case .recordingQuestion:
                if let range = self.fullTranscriptLower.range(of: self.wakeWord, options: .backwards) {
                    let afterWake = text[range.upperBound...].trimmingCharacters(in: .whitespaces)
                    if afterWake != self.lastQuestionSlice {
                        self.lastQuestionSlice = afterWake
                        self.lastChangeAt = Date()
                        self.scheduleSilenceTimer()
                    }
                }

            case .idle:
                break
            }

            if result.isFinal {
                if self.state == .recordingQuestion && !self.lastQuestionSlice.isEmpty {
                    self.finishWithQuestion(self.lastQuestionSlice)
                } else {
                    self.stop()
                }
            }
        }
    }

    // Silence handling
    private func scheduleSilenceTimer() {
        invalidateSilenceTimer()
        silenceTimer = Timer.scheduledTimer(withTimeInterval: 0.2, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            if Date().timeIntervalSince(self.lastChangeAt) >= self.silenceWindow,
               self.state == .recordingQuestion,
               !self.lastQuestionSlice.isEmpty {
                self.finishWithQuestion(self.lastQuestionSlice)
            }
        }
        RunLoop.main.add(silenceTimer!, forMode: .common)
    }

    private func invalidateSilenceTimer() {
        silenceTimer?.invalidate()
        silenceTimer = nil
    }

    private func finishWithQuestion(_ q: String) {
        DispatchQueue.main.async { self.lastQuestion = q }
        stop() // release mic; caller restarts after TTS if desired
    }
}
