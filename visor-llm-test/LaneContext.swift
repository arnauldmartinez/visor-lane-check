//
//  LaneContext.swift
//  visor-llm-test
//
//  Created by Arnauld Martinez on 11/8/25.
//

import Foundation

/// In-app, thread-safe store for the most recent lane inference.
/// No cross-process or app-group sharing.
public final class LaneContext {

    public static let shared = LaneContext()

    private let q = DispatchQueue(label: "lane.context.queue", attributes: .concurrent)
    private var _ego: Int? = nil
    private var _total: Int? = nil

    private init() {}

    /// Update from your lane inference (call from *this* app).
    public func update(ego: Int, total: Int) {
        let e = max(1, ego)
        let t = max(0, total)
        q.async(flags: .barrier) {
            self._ego = e
            self._total = t
        }
    }

    /// Read current (if available).
    public func snapshot() -> (ego: Int, total: Int)? {
        var s: (Int, Int)?
        q.sync {
            if let e = _ego, let t = _total { s = (e, t) }
        }
        return s
    }

    /// Pre-built context line for prompts, or nil if not set yet.
    public func contextLine() -> String? {
        if let s = snapshot() {
            return "Context: User is currently in lane \(s.ego) out of \(s.total). The user should be in the far left lane, N=1"
        }
        return nil
    }
}
