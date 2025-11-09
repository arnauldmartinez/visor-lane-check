import SwiftUI

struct ContentView: View {
    // LLM + wake
    @StateObject private var vm = LLMViewModel()
    @StateObject private var wake = SpeechWakeService(wakeWord: "visor", silenceWindow: 0.35)
    @StateObject private var speaker = Speaker()

    // Lane vision
    @StateObject private var lane = LaneVM()

    @State private var wakeEnabled = false
    @State private var manualPrompt = ""
    @State private var showDebug = true

    var body: some View {
        GeometryReader { geo in
            HStack(spacing: 12) {
                // ========== LEFT: LLM panel ==========
                VStack(spacing: 12) {
                    Text("On-Device LLM + “Visor” Wake").font(.title2).bold()

                    HStack {
                        Button {
                            Task { await vm.loadModelIfNeeded() }
                        } label: {
                            Label(vm.isDownloading ? "Loading…" : "Load Model", systemImage: "square.and.arrow.down")
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(vm.isGenerating || vm.isDownloading)

                        if vm.isDownloading {
                            ProgressView(value: vm.downloadProgress)
                                .frame(width: 120)
                                .padding(.leading, 8)
                        }

                        Spacer()
                        Toggle("Debug", isOn: $showDebug).toggleStyle(.switch)
                    }

                    Toggle(isOn: $wakeEnabled) {
                        Label("Enable “visor” wake (on-device)", systemImage: "mic.fill")
                    }
                    .onChange(of: wakeEnabled) { _, on in
                        if on { wake.onWake = {}; wake.start() } else { wake.stop() }
                    }

                    if showDebug {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Wake state: \(wake.state.rawValue)").font(.footnote)
                            if !wake.lastPartial.isEmpty {
                                Text("Partial: \(wake.lastPartial)")
                                    .font(.system(size: 12, weight: .regular, design: .monospaced))
                                    .lineLimit(3)
                            }
                            if !wake.lastQuestion.isEmpty {
                                Text("Question: \(wake.lastQuestion)")
                                    .font(.system(size: 12, weight: .bold, design: .monospaced))
                            }
                            Text("TTS speaking: \(speaker.isSpeaking ? "Yes" : "No")")
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        }
                        .padding(8)
                        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
                    }

                    TextEditor(text: $manualPrompt)
                        .frame(minHeight: 80)
                        .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
                        .onAppear { manualPrompt = "Explain what a Hamiltonian is in one sentence." }

                    HStack(spacing: 8) {
                        Button {
                            speaker.stop()
                            vm.generate(for: manualPrompt)
                        } label: {
                            Label(vm.isGenerating ? "Generating…" : "Run Locally", systemImage: "bolt.fill")
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(vm.isDownloading || vm.isGenerating)

                        Button(role: .destructive) {
                            vm.cancelGeneration()
                            speaker.stop()
                        } label: { Label("Cancel", systemImage: "xmark.circle.fill") }
                        .disabled(!vm.isGenerating)

                        Spacer()
                        if !wake.lastQuestion.isEmpty {
                            Text("Heard: “\(wake.lastQuestion)”")
                                .font(.footnote).foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                    }

                    // ===== Output + context used =====
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Output").font(.headline)
                        ScrollView {
                            VStack(alignment: .leading, spacing: 6) {
                                Text(vm.output.isEmpty ? "—" : vm.output)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .textSelection(.enabled)
                                    .padding(8)
                                    .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 8))

                                if !vm.decisionContextText.isEmpty {
                                    Text(vm.decisionContextText)
                                        .font(.footnote.monospaced())
                                        .foregroundStyle(.secondary)
                                }
                            }
                        }
                        .frame(minHeight: 120)
                    }

                    VStack(alignment: .leading, spacing: 2) {
                        Text("Timings (ms)").font(.headline)
                        Text(String(format: "Prepare: %.0f | First token: %.0f | Gen: %.0f | Total: %.0f",
                                    vm.prepareMS, vm.firstTokenMS, vm.genMS, vm.totalMS))
                        Text(String(format: "Tokens: %d  |  %.1f tok/s", vm.tokenCount, vm.tokensPerSec))
                    }
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)

                    Spacer()
                }
                .frame(width: geo.size.width * 0.45)
                .padding(.leading, 8)

                // ========== RIGHT: Camera + overlay ==========
                VStack(spacing: 8) {
                    ZStack {
                        CameraPreview(session: lane.cam.session)
                            .overlay(
                                Group {
                                    if lane.overlayEnabled, let img = lane.latestOverlay {
                                        Image(uiImage: img)
                                            .resizable()
                                            .scaledToFill()
                                            .opacity(0.9)
                                    }
                                }
                            )
                            .clipShape(RoundedRectangle(cornerRadius: 12))

                        VStack {
                            Spacer()
                            HStack {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(lane.laneReport).font(.footnote.monospaced())
                                        .padding(6).background(.ultraThinMaterial)
                                        .clipShape(RoundedRectangle(cornerRadius: 6))
                                    Text(lane.timingReport).font(.caption2.monospaced())
                                        .foregroundStyle(.secondary)
                                        .padding(6).background(.ultraThinMaterial)
                                        .clipShape(RoundedRectangle(cornerRadius: 6))
                                }
                                Spacer()
                            }
                            .padding(.horizontal, 12)
                            .padding(.bottom, 8)
                        }
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)

                    HStack(spacing: 10) {
                        Toggle("Overlay", isOn: $lane.overlayEnabled).buttonStyle(.automatic)
                        Spacer()
                        Button("Start Camera") { lane.start() }.buttonStyle(.borderedProminent)
                        Button("Stop Camera", role: .destructive) { lane.stop() }.buttonStyle(.bordered)
                    }
                }
                .frame(width: geo.size.width * 0.55)
                .padding(.trailing, 8)
            }
        }
        .ignoresSafeArea()
        .onChange(of: wake.lastQuestion) { _, question in
            let q = question.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !q.isEmpty else { return }
            speaker.stop()
            vm.generate(for: q)
        }
        .onChange(of: vm.isGenerating) { _, nowGen in
            if nowGen == false, !vm.output.isEmpty { speaker.speak(vm.output) }
        }
        .onChange(of: speaker.isSpeaking) { _, speaking in
            if speaking == false, wakeEnabled, wake.state == .idle {
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) {
                    if self.wakeEnabled && self.wake.state == .idle { self.wake.start() }
                }
            }
        }
        .onAppear {
            lane.start()
        }
        .onDisappear {
            if wakeEnabled { wake.stop() }
            speaker.stop()
            lane.stop()
        }
    }
}
