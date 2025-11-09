import Foundation
import Combine
import QuartzCore          // CACurrentMediaTime()
import MLXLLM
import MLXLMCommon
import Tokenizers

@MainActor
final class LLMViewModel: ObservableObject {
    @Published var status: String = "Idle"
    @Published var isDownloading: Bool = false
    @Published var downloadProgress: Double = 0.0
    @Published var isGenerating: Bool = false
    @Published var output: String = ""

    // Visible context used for this specific response (for UI)
    @Published var decisionContextText: String = ""   // e.g., "Context used: lane 3 of 5 (N=1 = far left)"

    // Timing (ms)
    @Published var totalMS: Double = 0
    @Published var prepareMS: Double = 0
    @Published var firstTokenMS: Double = 0
    @Published var genMS: Double = 0
    @Published var tokenCount: Int = 0
    @Published var tokensPerSec: Double = 0

    private var firstTokenAt: CFTimeInterval? = nil
    private var streamedTokenCount: Int = 0

    private var container: ModelContainer?
    private var genTask: Task<Void, Never>?

    private let MAX_INPUT_CHARS = 2000

    // === Updated system prompt: clear rule (correct lane = 1 of N) and required behavior ===
    private let SYSTEM_PREFIX = """
You are an on-device driving assistant. You receive a lane context and a user question.

Lane semantics:
- Lanes are numbered from LEFT to RIGHT as 1..N.
- "Correct lane" means lane 1 of N (the far-left lane).
- You will be given: ego lane (E) and total lanes (N).

On every answer:
1) Include a context line: "Lane context: E of N (1 = far-left)".
2) If the question is about lane correctness or implies lane choice, explicitly state:
   - "You are in the correct lane."       if E == 1
   - "You are NOT in the correct lane; move left when safe to be in lane 1." if E != 1
3) Otherwise, answer normally but still include the context line.
4) Be concise (≤ 2 short sentences unless the user asks for more).
"""

    private let configuration: ModelConfiguration = LLMRegistry.llama3_2_1B_4bit

    func loadModelIfNeeded() async {
        guard container == nil else { status = "Model ready"; return }
        #if targetEnvironment(simulator)
        status = "MLX requires a real device (Simulator lacks proper Metal GPU family)."
        return
        #endif

        status = "Loading model…"
        isDownloading = true
        downloadProgress = 0

        do {
            let c = try await LLMModelFactory.shared.loadContainer(configuration: configuration) { progress in
                Task { @MainActor in self.downloadProgress = progress.fractionCompleted }
            }
            container = c
            status = "Model ready"
        } catch {
            status = "Load failed: \(error.localizedDescription)"
        }
        isDownloading = false
    }

    func cancelGeneration() { genTask?.cancel() }

    func generate(for userPrompt: String) {
        guard let container else { status = "Model not loaded"; return }
        let trimmed = userPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { status = "Say or type a question"; return }

        // Capture the lane snapshot used for this response (for UI display)
        let snap = LaneContext.shared.snapshot()
        if let s = snap {
            decisionContextText = "Context used: lane \(s.ego) of \(s.total) (N=1 = far left)"
        } else {
            decisionContextText = "Context used: (none yet)"
        }

        // Build the explicit lane context line we feed to the model
        let laneCtxLine: String = {
            if let s = snap {
                return "Lane context: E=\(s.ego), N=\(s.total) (1 = far-left)"
            } else {
                return "Lane context: unavailable"
            }
        }()

        // Final prompt: system rules + lane context line + user question
        let finalPrompt = """
        \(SYSTEM_PREFIX)

        \(laneCtxLine)
        User: \(trimmed)
        """

        guard finalPrompt.count <= MAX_INPUT_CHARS else {
            status = "Error: Prompt too long (\(finalPrompt.count) > \(MAX_INPUT_CHARS))."
            return
        }

        // Reset metrics and state
        output = ""
        tokenCount = 0
        prepareMS = 0
        firstTokenMS = 0
        genMS = 0
        totalMS = 0
        tokensPerSec = 0
        firstTokenAt = nil
        streamedTokenCount = 0

        isGenerating = true
        status = "Generating…"

        let tStart = CACurrentMediaTime()

        genTask?.cancel()
        genTask = Task { [weak self] in
            guard let self else { return }
            do {
                _ = try await container.perform { context in
                    // Prepare (timed)
                    let t0 = CACurrentMediaTime()
                    let input = try await context.processor.prepare(input: UserInput(prompt: finalPrompt))
                    let t1 = CACurrentMediaTime()
                    Task { @MainActor in self.prepareMS = (t1 - t0) * 1000.0 }

                    // Params
                    let params = GenerateParameters(maxTokens: 256, temperature: 0.7, topP: 0.95)

                    // Stream tokens → update on main
                    return try MLXLMCommon.generate(input: input, parameters: params, context: context) { tokens in
                        guard let lastID = tokens.last else { return .more }
                        let now = CACurrentMediaTime()
                        let piece = context.tokenizer.decode(tokens: [lastID])
                        Task { @MainActor in
                            if self.firstTokenAt == nil {
                                self.firstTokenAt = now
                                self.firstTokenMS = (now - tStart) * 1000.0
                            }
                            if self.output.hasSuffix(piece) == false { self.output += piece }
                            self.streamedTokenCount += 1
                            self.tokenCount = self.streamedTokenCount
                        }
                        if Task.isCancelled { return .stop }
                        return .more
                    }
                }

                let tEnd = CACurrentMediaTime()
                await MainActor.run {
                    let total = (tEnd - tStart) * 1000.0
                    let genOnly = (self.firstTokenAt != nil) ? (tEnd - self.firstTokenAt!) * 1000.0 : 0
                    let tps = genOnly > 0 ? Double(self.streamedTokenCount) / (genOnly / 1000.0) : 0
                    self.genMS = genOnly
                    self.totalMS = total
                    self.tokensPerSec = tps
                    self.status = "Done"
                    self.isGenerating = false
                }
            } catch {
                await MainActor.run {
                    self.status = Task.isCancelled ? "Cancelled" : "Error: \(error.localizedDescription)"
                    self.isGenerating = false
                }
            }
        }
    }
}
