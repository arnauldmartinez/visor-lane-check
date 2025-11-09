import SwiftUI
import Combine            // ← FIX: needed for ObservableObject/@Published/@StateObject
import MapKit
import CoreLocation

// ===== Local overlay helper for drawing a polyline route =====
struct RouteOverlay: MapContent {
    let route: MKRoute
    var body: some MapContent {
        MapPolyline(route.polyline).stroke(.blue, lineWidth: 6)
    }
}

// ===== Minimal NavManager for routing on the Map =====
final class NavManager: NSObject, ObservableObject, CLLocationManagerDelegate {
    @Published var route: MKRoute?
    @Published var userLocation: CLLocation?

    private let loc = CLLocationManager()

    override init() {
        super.init()
        loc.delegate = self
        loc.desiredAccuracy = kCLLocationAccuracyBest
        loc.requestWhenInUseAuthorization()
        loc.startUpdatingLocation()
    }

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        userLocation = locations.last
    }

    func buildRoute(from: MKMapItem, to: MKMapItem) {
        let req = MKDirections.Request()
        req.source = from
        req.destination = to
        req.transportType = .automobile
        MKDirections(request: req).calculate { [weak self] rsp, err in
            guard let self, let r = rsp?.routes.first, err == nil else { return }
            DispatchQueue.main.async { self.route = r }
        }
    }
}

struct ContentView: View {
    // ===== LLM + wake (auto-load + auto-listen) =====
    @StateObject private var vm = LLMViewModel()
    @StateObject private var wake = SpeechWakeService(wakeWord: "visor", silenceWindow: 0.35)
    @StateObject private var speaker = Speaker()

    // ===== Lane vision (camera auto-start, raw feed only) =====
    @StateObject private var lane = LaneVM()

    // ===== Navigation for Map =====
    @StateObject private var nav = NavManager()

    // Map camera
    @State private var mapCamera: MapCameraPosition = .userLocation(fallback: .automatic)
    @State private var locationAuthRequested = false

    // Wake always on
    @State private var wakeEnabled = true

    // Routing inputs
    @State private var startQuery: String = ""
    @State private var endQuery: String = ""
    @State private var isRouting: Bool = false
    @State private var routeError: String?

    var body: some View {
        GeometryReader { geo in
            HStack(spacing: 12) {

                // ================= LEFT: MAP (dominant) =================
                VStack(spacing: 8) {
                    // Routing input bar
                    HStack(spacing: 8) {
                        TextField("Start (address or lat,lon)", text: $startQuery)
                            .textInputAutocapitalization(.never)
                            .disableAutocorrection(true)
                            .textFieldStyle(.roundedBorder)

                        TextField("End (address or lat,lon)", text: $endQuery)
                            .textInputAutocapitalization(.never)
                            .disableAutocorrection(true)
                            .textFieldStyle(.roundedBorder)

                        Button {
                            Task { await buildRouteFromQueries() }
                        } label: {
                            if isRouting {
                                ProgressView().frame(width: 22, height: 22)
                            } else {
                                Text("Route")
                            }
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(isRouting || startQuery.isEmpty || endQuery.isEmpty)
                    }

                    if let err = routeError {
                        Text(err).font(.footnote).foregroundStyle(.red)
                    }

                    ZStack {
                        Map(
                            position: $mapCamera,
                            interactionModes: [.pan, .zoom, .rotate],
                            content: {
                                if let r = nav.route { RouteOverlay(route: r) }
                                UserAnnotation()
                            }
                        )
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
                .frame(width: geo.size.width * 0.60, height: geo.size.height)
                .padding(.leading, 8)

                // ================= RIGHT: CAMERA (raw feed) + HUD =================
                VStack(spacing: 8) {
                    ZStack {
                        // Raw camera only (no overlay)
                        CameraPreview(session: lane.cam.session)
                            .clipShape(RoundedRectangle(cornerRadius: 12))

                        // Lane + timing HUD + exact lane context used by LLM
                        VStack {
                            Spacer()
                            HStack {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(lane.laneReport)
                                        .font(.footnote.monospaced())
                                        .padding(6)
                                        .background(.ultraThinMaterial)
                                        .clipShape(RoundedRectangle(cornerRadius: 6))

                                    Text(lane.timingReport)
                                        .font(.caption2.monospaced())
                                        .foregroundStyle(.secondary)
                                        .padding(6)
                                        .background(.ultraThinMaterial)
                                        .clipShape(RoundedRectangle(cornerRadius: 6))

                                    if !vm.decisionContextText.isEmpty {
                                        Text(vm.decisionContextText)
                                            .font(.caption2.monospaced())
                                            .foregroundStyle(.secondary)
                                            .padding(.horizontal, 6)
                                            .padding(.vertical, 4)
                                            .background(.ultraThinMaterial)
                                            .clipShape(RoundedRectangle(cornerRadius: 6))
                                    }
                                }
                                Spacer()
                            }
                            .padding(.horizontal, 12)
                            .padding(.bottom, 8)
                        }
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
                .frame(width: geo.size.width * 0.40, height: geo.size.height)
                .padding(.trailing, 8)
            }
        }
        .ignoresSafeArea()

        // ===== Auto behaviors (unchanged logic) =====
        .task {
            await vm.loadModelIfNeeded()
            if wakeEnabled { wake.onWake = {}; wake.start() }
        }
        .onAppear {
            lane.overlayEnabled = false   // ensure raw feed only
            lane.start()                  // auto start camera/inference
            requestLocationIfNeeded()
        }
        .onDisappear {
            if wakeEnabled { wake.stop() }
            speaker.stop()
            lane.stop()
        }

        // Voice → LLM
        .onChange(of: wake.lastQuestion) { _, question in
            let q = question.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !q.isEmpty else { return }
            speaker.stop()
            vm.generate(for: q)
        }

        // Speak when finished
        .onChange(of: vm.isGenerating) { _, nowGen in
            if nowGen == false, !vm.output.isEmpty { speaker.speak(vm.output) }
        }

        // Re-arm wake after TTS completes
        .onChange(of: speaker.isSpeaking) { _, speaking in
            if speaking == false, wakeEnabled, wake.state == .idle {
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) {
                    if self.wakeEnabled && self.wake.state == .idle { self.wake.start() }
                }
            }
        }
    }

    // ===== helpers =====

    private func requestLocationIfNeeded() {
        guard !locationAuthRequested else { return }
        locationAuthRequested = true
        CLLocationManager().requestWhenInUseAuthorization()
    }

    private func makeItem(_ coord: CLLocationCoordinate2D, name: String) -> MKMapItem {
        let placemark = MKPlacemark(coordinate: coord)
        let item = MKMapItem(placemark: placemark)
        item.name = name
        return item
    }

    private func parseLatLon(_ s: String) -> CLLocationCoordinate2D? {
        // Accept "lat,lon" (with optional spaces)
        let parts = s.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
        guard parts.count == 2, let lat = Double(parts[0]), let lon = Double(parts[1]) else { return nil }
        guard abs(lat) <= 90, abs(lon) <= 180 else { return nil }
        return CLLocationCoordinate2D(latitude: lat, longitude: lon)
    }

    private func localSearchMapItem(for query: String) async throws -> MKMapItem {
        if let coord = parseLatLon(query) {
            return makeItem(coord, name: query)
        }
        let req = MKLocalSearch.Request()
        req.naturalLanguageQuery = query
        let search = MKLocalSearch(request: req)
        let rsp = try await search.start()
        if let item = rsp.mapItems.first { return item }
        throw NSError(domain: "Route", code: -1, userInfo: [NSLocalizedDescriptionKey: "No results for '\(query)'"])
    }

    private func buildRouteFromQueries() async {
        guard !startQuery.isEmpty, !endQuery.isEmpty else { return }
        isRouting = true
        routeError = nil
        do {
            let startItem = try await localSearchMapItem(for: startQuery)
            let endItem   = try await localSearchMapItem(for: endQuery)
            await MainActor.run { nav.buildRoute(from: startItem, to: endItem) }
        } catch {
            await MainActor.run { self.routeError = (error as NSError).localizedDescription }
        }
        isRouting = false
    }
}
