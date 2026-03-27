import Foundation

nonisolated struct RecommendationMessage: Codable, Sendable {
    let plantId: Int
    let processedPlantId: Int?
    let disease: String
    let text: String
}

@MainActor
final class PlantWebSocketService {
    static let shared = PlantWebSocketService()
    private init() {}

    var onPlantReceived: ((Plant) -> Void)?
    var onRecommendationReceived: ((Int, Int?, String, String) -> Void)?

    private var webSocketTask: URLSessionWebSocketTask?
    private var isConnected = false
    private var stompConnected = false  // True only after CONNECTED frame received

    private lazy var urlSession: URLSession = {
        let config = URLSessionConfiguration.default
        config.waitsForConnectivity = true
        return URLSession(configuration: config)
    }()

    private var currentUserId: Int?
    private var reconnectTask: Task<Void, Never>?
    private var receiveLoopTask: Task<Void, Never>?
    private var heartbeatTask: Task<Void, Never>?
    private let reconnectDelay: TimeInterval = 5

    func connect(userId: Int) {
        currentUserId = userId

        guard !isConnected else {
            print("[WS] ⚠️ Already connected, skipping connect()")
            return
        }

        Task {
            guard let token = await APIClient.shared.getToken() else {
                print("[WS] ❌ No JWT token available — skipping WebSocket connect")
                return
            }

            let wsURL = AppConfig.wsURL
            guard let url = URL(string: wsURL) else {
                print("[WS] ❌ Invalid WebSocket URL: \(wsURL)")
                return
            }

            print("[WS] 🔵 Connecting to \(url) for userId: \(userId)")

            webSocketTask?.cancel(with: .goingAway, reason: nil)

            webSocketTask = urlSession.webSocketTask(with: url)
            webSocketTask?.resume()
            isConnected = true
            stompConnected = false

            let host = url.host ?? "morzio.com"
            let connectFrame = StompFrame.connect(host: host, token: token)
            print("[WS] 📤 Sending CONNECT frame")
            await sendAsync(frame: connectFrame)

            startReceiveLoop()
        }
    }


    func disconnect() {
        print("[WS] Disconnecting...")
        reconnectTask?.cancel()
        reconnectTask = nil
        heartbeatTask?.cancel()
        heartbeatTask = nil
        receiveLoopTask?.cancel()
        receiveLoopTask = nil

        if stompConnected {
            let frame = StompFrame.disconnect()
            webSocketTask?.send(.string(frame)) { _ in }
        }

        webSocketTask?.cancel(with: .goingAway, reason: nil)
        webSocketTask = nil
        isConnected = false
        stompConnected = false
        print("[WS] Disconnected")
    }

    private func startReceiveLoop() {
        receiveLoopTask?.cancel()
        receiveLoopTask = Task { [weak self] in
            guard let self else { return }
            print("[WS] 🟢 Receive loop started")

            while !Task.isCancelled {
                guard let task = self.webSocketTask else {
                    print("[WS] ❌ Receive loop: webSocketTask is nil, exiting")
                    break
                }

                do {
                    let message = try await task.receive()
                    await self.handleMessage(message)
                } catch {
                    if Task.isCancelled { 
                        print("[WS] 🟡 Receive loop cancelled by task")
                        break 
                    }
                    print("[WS] ❌ Receive error: \(error)")
                    print("[WS] ❌ Error type: \(type(of: error))")
                    print("[WS] ❌ Error description: \(error.localizedDescription)")
                    
                    if self.isConnected {
                        self.isConnected = false
                        self.stompConnected = false
                        print("[WS] 🔄 Scheduling reconnect due to error...")
                        self.scheduleReconnect()
                    }
                    break
                }
            }

            print("[WS] 👂 Receive loop ended")
        }
    }

    private func handleMessage(_ message: URLSessionWebSocketTask.Message) async {
        switch message {
        case .string(let text):
            if StompFrame.isHeartbeat(text) {
                print("[WS] Heartbeat received")
                return
            }

            let displayText = text.replacingOccurrences(of: "\0", with: "\\0")
            print("[WS] RAW FRAME:\n\(displayText)")

            guard let command = StompFrame.command(from: text) else {
                print("[WS] Could not parse command from frame")
                return
            }

            print("[WS] Command: \(command)")

            switch command {
            case "CONNECTED":
                handleConnected(raw: text)

            case "MESSAGE":
                handleMessageFrame(raw: text)

            case "ERROR":
                print("[WS] STOMP ERROR frame:\n\(displayText)")
                if let errorBody = StompFrame.body(from: text) {
                    print("[WS] Error body: \(errorBody)")
                }

            case "RECEIPT":
                print("[WS] RECEIPT received")

            default:
                print("[WS] Unknown STOMP command: \(command)")
            }

        case .data(let data):
            print("[WS] Binary data received (\(data.count) bytes) — ignoring")

        @unknown default:
            print("[WS] Unknown message type received")
        }
    }


    private func handleConnected(raw: String) {
        stompConnected = true
        print("[WS] STOMP CONNECTED")

        if let serverHeartbeat = StompFrame.header("heart-beat", from: raw) {
            print("[WS] Server heart-beat: \(serverHeartbeat)")
        }
        if let version = StompFrame.header("version", from: raw) {
            print("[WS] STOMP version: \(version)")
        }

        guard let userId = currentUserId else {
            print("[WS] No userId available, cannot subscribe")
            return
        }


        let userDest = "/user/queue/plants"
        let subFrame1 = StompFrame.subscribe(destination: userDest, id: "sub-user-plants")
        print("[WS] SUBSCRIBE to \(userDest)")
        Task { await sendAsync(frame: subFrame1) }

        let explicitDest = "/user/\(userId)/queue/plants"
        let subFrame2 = StompFrame.subscribe(destination: explicitDest, id: "sub-explicit-plants")
        print("[WS] SUBSCRIBE to \(explicitDest)")
        Task { await sendAsync(frame: subFrame2) }

        let topicDest = "/topic/plants"
        let subFrame3 = StompFrame.subscribe(destination: topicDest, id: "sub-topic-plants")
        print("[WS] SUBSCRIBE to \(topicDest)")
        Task { await sendAsync(frame: subFrame3) }

        // Recommendations — user queue
        let recoUserDest = "/user/queue/recommendations"
        let recoFrame1 = StompFrame.subscribe(destination: recoUserDest, id: "sub-user-recommendations")
        print("[WS] SUBSCRIBE to \(recoUserDest)")
        Task { await sendAsync(frame: recoFrame1) }

        let recoExplicitDest = "/user/\(userId)/queue/recommendations"
        let recoFrame2 = StompFrame.subscribe(destination: recoExplicitDest, id: "sub-explicit-recommendations")
        print("[WS] SUBSCRIBE to \(recoExplicitDest)")
        Task { await sendAsync(frame: recoFrame2) }

        let recoTopicDest = "/topic/recommendations"
        let recoFrame3 = StompFrame.subscribe(destination: recoTopicDest, id: "sub-topic-recommendations")
        print("[WS] SUBSCRIBE to \(recoTopicDest)")
        Task { await sendAsync(frame: recoFrame3) }

        startHeartbeat()
    }


    private func handleMessageFrame(raw: String) {
        let destination = StompFrame.header("destination", from: raw)
        print("[WS] MESSAGE destination: \(destination ?? "unknown")")

        guard let body = StompFrame.body(from: raw) else {
            print("[WS] MESSAGE frame has no body")
            return
        }

        print("[WS] MESSAGE body: \(body)")

        guard let data = body.data(using: .utf8) else {
            print("[WS] Failed to convert body to UTF-8 data")
            return
        }

        // Route by destination
        if destination?.contains("/queue/recommendations") == true
            || destination?.contains("/topic/recommendations") == true {
            handleRecommendationBody(data: data, body: body)
        } else {
            handlePlantBody(data: data, body: body)
        }
    }

    private func handlePlantBody(data: Data, body: String) {
        do {
            let plant = try JSONDecoder().decode(Plant.self, from: data)
            print("[WS] Decoded plant: id=\(plant.id), lat=\(plant.latitude), lon=\(plant.longitude)")

            if let callback = onPlantReceived {
                callback(plant)
                print("[WS] onPlantReceived callback fired")
            } else {
                print("[WS] onPlantReceived callback is nil — no listener attached!")
            }
        } catch {
            print("[WS] Failed to decode Plant from body: \(error)")
            print("[WS] Raw body was: \(body)")

            if let envelope = try? JSONDecoder().decode(APIEnvelope<Plant>.self, from: data),
               let plant = envelope.data {
                print("[WS] Decoded plant from envelope: id=\(plant.id)")
                onPlantReceived?(plant)
            }
        }
    }

    private func handleRecommendationBody(data: Data, body: String) {
        do {
            let msg = try JSONDecoder().decode(RecommendationMessage.self, from: data)
            print("[WS] Recommendation received for plant \(msg.plantId): \(msg.disease)")
            
            // Always persist the recommendation data, even if no listener is attached
            UserDefaults.standard.set(msg.disease, forKey: "plant_\(msg.plantId)_disease")
            UserDefaults.standard.set(msg.text, forKey: "plant_\(msg.plantId)_recommendation")
            if let processedPlantId = msg.processedPlantId {
                UserDefaults.standard.set(processedPlantId, forKey: "plant_\(msg.plantId)_processed_plant_id")
            }
            print("[WS] Cached recommendation for plant \(msg.plantId)")

            if let callback = onRecommendationReceived {
                callback(msg.plantId, msg.processedPlantId, msg.disease, msg.text)
                print("[WS] onRecommendationReceived callback fired")
            } else {
                print("[WS] onRecommendationReceived callback is nil — recommendation cached for later")
            }
        } catch {
            print("[WS] Failed to decode recommendation: \(error)")
            print("[WS] Raw body was: \(body)")
        }
    }


    private func startHeartbeat() {
        heartbeatTask?.cancel()
        heartbeatTask = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 10_000_000_000) // 10 seconds
                guard !Task.isCancelled else { break }
                guard let self, self.stompConnected else { break }

                self.webSocketTask?.send(.string("\n")) { error in
                    if let error {
                        print("[WS] Heartbeat send error: \(error)")
                    }
                }
            }
        }
    }


    private func sendAsync(frame: String) async {
        guard let task = webSocketTask else {
            print("[WS] Cannot send — webSocketTask is nil")
            return
        }
        do {
            try await task.send(.string(frame))
        } catch {
            print("[WS] Send error: \(error)")
        }
    }


    private func scheduleReconnect() {
        reconnectTask?.cancel()
        heartbeatTask?.cancel()
        receiveLoopTask?.cancel()
        webSocketTask?.cancel(with: .goingAway, reason: nil)
        webSocketTask = nil

        print("[WS] Scheduling reconnect in \(reconnectDelay)s")
        reconnectTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(5 * 1_000_000_000))
            guard !Task.isCancelled else { return }
            guard let self else { return }
            guard let userId = self.currentUserId else { return }
            self.isConnected = false
            self.stompConnected = false
            print("[WS] Reconnecting now...")
            self.connect(userId: userId)
        }
    }
}
