//
//  LaneVM.swift
//  visor-llm-test
//
//  Created by Arnauld Martinez on 11/8/25.
//

import Foundation
import UIKit
import CoreML
import Combine   // <-- add

@MainActor
final class LaneVM: ObservableObject {
    @Published var latestOverlay: UIImage?
    @Published var laneReport: String = "—"
    @Published var timingReport: String = "—"
    @Published var overlayEnabled: Bool = false

    let cam = CameraManager()
    private var mlModel: MLModel?
    private var timer: DispatchSourceTimer?
    private var active = false

    // knobs
    private let targetCanvas = CGSize(width: 1280, height: 720)
    private let frameInterval: TimeInterval = 0.25 // ~ 4 FPS

    init() {}

    func start() {
        do { try ModelManager.shared.loadIfNeeded(); mlModel = ModelManager.shared.model }
        catch { print("Model load error:", error) }
        cam.onFrame = { _ in /* pulled by timer */ }
        cam.start()
        setActive(true)
    }

    func stop() {
        setActive(false)
        cam.stop()
        latestOverlay = nil
    }

    func setActive(_ on: Bool) {
        active = on
        if on { startTimer() } else { stopTimer() }
    }

    private func startTimer() {
        guard timer == nil else { return }
        let t = DispatchSource.makeTimerSource(queue: .global(qos: .userInitiated))
        t.schedule(deadline: .now() + 0.2, repeating: frameInterval)
        t.setEventHandler { [weak self] in self?.tick() }
        t.resume()
        timer = t
    }
    private func stopTimer() { timer?.cancel(); timer = nil }

    private func tick() {
        guard active, let ml = mlModel, let cg = cam.latestFrame else { return }

        func now() -> CFAbsoluteTime { CFAbsoluteTimeGetCurrent() }
        func ms(_ s: CFAbsoluteTime) -> Double { s * 1000.0 }
        func fmt(_ x: Double) -> String { String(format: "%.1f", x) }

        let tTotal0 = now()
        let inputSize = InferenceUtils.requiredInputSize(from: ml) ?? CGSize(width: 640, height: 640)

        // --- preprocess on main ---
        var im0: UIImage!
        var provider: MLFeatureProvider!
        var padTop = 0, padBottom = 0

        let tPre0 = now()
        DispatchQueue.main.sync {
            let ui = UIImage(cgImage: cg)
            guard let canvas = ui.resizedToCanvas(targetCanvas) else { return }
            im0 = canvas
            let lb = try! UtilsBridge.letterbox(image: canvas, targetSize: inputSize)
            padTop = lb.padTop; padBottom = lb.padBottom
            provider = try! self.inputProvider(for: ml, image: lb.image)
        }
        let tPre = now() - tPre0

        // --- inference ---
        let tInf0 = now()
        let out: MLFeatureProvider
        do { out = try ml.prediction(from: provider) } catch { return }
        let tInf = now() - tInf0

        // --- postproc ---
        let tPost0 = now()
        guard let (daML, llML) = try? InferenceUtils.pickSegHeads(from: out),
              let (daMask, llMask) = try? UtilsFast.makeMasksFast(drivable: daML, lane: llML, padTop: padTop, padBottom: padBottom) else { return }

        let pOut = UtilsBridge.parabolaLaneCount(daMask: daMask, llMask: llMask, egoX: -1, buildPath: overlayEnabled)
        var vis: UIImage?
        if overlayEnabled {
            DispatchQueue.main.sync {
                if let base = try? UtilsBridge.showSegResult(baseBGR: im0, drivableMask: daMask, laneMask: llMask),
                   let over = try? UtilsBridge.overlayParabola(on: base, parabola: pOut.path, egoLane: pOut.egoLane, totalLanes: pOut.totalLanes) {
                    vis = over
                }
            }
        }
        let tPost = now() - tPost0
        let tTotal = now() - tTotal0

        // --- UI + LaneContext ---
        DispatchQueue.main.async {
            self.latestOverlay = vis
            self.laneReport = "Ego lane: \(pOut.egoLane)\nTotal lanes: \(pOut.totalLanes)"
            self.timingReport =
"""
Round trip: \(fmt(ms(tTotal))) ms
  • preproc:   \(fmt(ms(tPre))) ms
  • inference: \(fmt(ms(tInf))) ms
  • postproc:  \(fmt(ms(tPost))) ms
"""
            // Publish into the in-app singleton for the LLM prompt
            LaneContext.shared.update(ego: pOut.egoLane, total: pOut.totalLanes)
        }
    }

    // provider must be on main (CG/UIGraphics)
    @MainActor
    private func inputProvider(for model: MLModel, image: UIImage) throws -> MLFeatureProvider {
        let fd = model.modelDescription.inputDescriptionsByName.values.first!
        if fd.type == .image {
            let pb = try image.cgImageOrDie().makeBGRA32PixelBuffer()
            return try MLDictionaryFeatureProvider(dictionary: [fd.name: pb])
        } else {
            let chw = image.toCHWFloat()
            let h = Int(image.size.height), w = Int(image.size.width)
            let arr = try MLMultiArray(shape: [1, 3, NSNumber(value: h), NSNumber(value: w)], dataType: .float32)
            chw.withUnsafeBufferPointer { src in
                let dst = UnsafeMutablePointer<Float32>(OpaquePointer(arr.dataPointer))
                dst.update(from: src.baseAddress!, count: src.count)
            }
            return try MLDictionaryFeatureProvider(dictionary: [fd.name: arr])
        }
    }
}

// ===== UIImage & CGImage helpers =====
private extension UIImage {
    func resizedToCanvas(_ canvas: CGSize) -> UIImage? {
        let r = min(canvas.width / size.width, canvas.height / size.height)
        let new = CGSize(width: size.width * r, height: size.height * r)
        let fmt = UIGraphicsImageRendererFormat()
        fmt.scale = scale; fmt.opaque = true
        return UIGraphicsImageRenderer(size: canvas, format: fmt).image { ctx in
            UIColor.black.setFill()
            ctx.fill(CGRect(origin: .zero, size: canvas))
            let x = (canvas.width - new.width) * 0.5
            let y = (canvas.height - new.height) * 0.5
            draw(in: CGRect(x: x, y: y, width: new.width, height: new.height))
        }
    }
    func toCHWFloat() -> [Float] {
        guard let cg = self.cgImage else { return [] }
        let W = cg.width, H = cg.height
        var rgba = [UInt8](repeating: 0, count: W * H * 4)
        let cs = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(data: &rgba, width: W, height: H, bitsPerComponent: 8, bytesPerRow: W * 4,
                            space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)!
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: W, height: H))
        var out = [Float](repeating: 0, count: 3 * W * H)
        for y in 0..<H {
            for x in 0..<W {
                let i = y * W + x
                let r = rgba[i*4 + 0], g = rgba[i*4 + 1], b = rgba[i*4 + 2]
                out[0*W*H + i] = Float(r) / 255.0
                out[1*W*H + i] = Float(g) / 255.0
                out[2*W*H + i] = Float(b) / 255.0
            }
        }
        return out
    }
    func cgImageOrDie() throws -> CGImage {
        guard let cg = self.cgImage else { throw NSError(domain: "cg", code: -1) }
        return cg
    }
}
private extension CGImage {
    func makeBGRA32PixelBuffer() throws -> CVPixelBuffer {
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        ]
        var pb: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, self.width, self.height,
                                         kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
        guard status == kCVReturnSuccess, let px = pb else { throw NSError(domain: "pb", code: -2) }
        CVPixelBufferLockBaseAddress(px, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(px, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(px) else { throw NSError(domain: "pb", code: -3) }
        let ctx = CGContext(data: base, width: self.width, height: self.height,
                            bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(px),
                            space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)!
        ctx.draw(self, in: CGRect(x: 0, y: 0, width: self.width, height: self.height))
        return px
    }
}
