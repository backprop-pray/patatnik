import SwiftUI
import SwiftUI
import Combine

private nonisolated struct RespondBody: Encodable, Sendable {
    let accepted: Bool
}

private nonisolated struct EmptyBody: Encodable, Sendable {}

private nonisolated struct RejectBody: Encodable, Sendable {
    let comment: String
}

private nonisolated struct UpdateRecommendationBody: Encodable, Sendable {
    let processedPlantId: Int
    let text: String
    
    enum CodingKeys: String, CodingKey {
        case processedPlantId = "processed_plant_id"
        case text
    }
}

private nonisolated struct UpdateRecommendationResponse: Decodable, Sendable {
    let processedPlantId: Int
    let plantId: Int
    let disease: String
    let text: String
    
    enum CodingKeys: String, CodingKey {
        case processedPlantId = "processed_plant_id"
        case plantId = "plant_id"
        case disease
        case text
    }
}

@MainActor
class PlantDetailViewModel: ObservableObject {

    // Recommendation
    @Published var estimatedDisease: String = ""
    @Published var recommendation: String = ""
    @Published var recommendationLoaded: Bool = false
    @Published var isLoadingRecommendation: Bool = true
    @Published var recommendationError: Bool = false

    // Response
    @Published var hasResponded: Bool = false
    @Published var responseAccepted: Bool = false

    // Opinion
    @Published var userOpinion: String = ""
    @Published var isSubmittingOpinion: Bool = false
    @Published var opinionSubmitted: Bool = false
    
    // Processed plant ID for PATCH endpoint
    private var processedPlantId: Int?

    private var loadingTask: Task<Void, Never>?
    private var currentPlantId: Int?

    func loadRecommendation(for plant: Plant) {
        currentPlantId = plant.id
        
        // Load persisted user response state
        let responseKey = "plant_\(plant.id)_responded"
        let acceptedKey = "plant_\(plant.id)_accepted"
        let opinionKey = "plant_\(plant.id)_opinion_submitted"
        
        hasResponded = UserDefaults.standard.bool(forKey: responseKey)
        responseAccepted = UserDefaults.standard.bool(forKey: acceptedKey)
        opinionSubmitted = UserDefaults.standard.bool(forKey: opinionKey)
        
        // Check if plant has recommendation data from backend
        if let disease = plant.disease, 
           let recommendedAction = plant.recommendedAction,
           !disease.isEmpty,
           !recommendedAction.isEmpty {
            
            // Use data from backend
            estimatedDisease = disease
            recommendation = recommendedAction
            
            // Get processedPlantId from backend data
            if let processedPlantId = plant.processedPlantId {
                self.processedPlantId = processedPlantId
                UserDefaults.standard.set(processedPlantId, forKey: "plant_\(plant.id)_processed_plant_id")
                print("[ViewModel] ✅ Loaded processedPlantId=\(processedPlantId) for plant \(plant.id)")
            } else {
                print("[ViewModel] ⚠️ WARNING: No processedPlantId in backend response for plant \(plant.id)")
                print("[ViewModel] Backend should include processedPlantId field!")
            }
            
            // Check status: nil or false = show accept/reject buttons, true = already responded
            if let status = plant.status, status == true {
                hasResponded = true
                responseAccepted = true  // status=true means accepted
            }
            
            recommendationLoaded = true
            isLoadingRecommendation = false
            recommendationError = false
            userOpinion = ""
            
            // Also cache it locally for offline use
            UserDefaults.standard.set(disease, forKey: "plant_\(plant.id)_disease")
            UserDefaults.standard.set(recommendedAction, forKey: "plant_\(plant.id)_recommendation")
            
            print("[ViewModel] Loaded recommendation from backend for plant \(plant.id)")
            return
        }
        
        // Try to load cached recommendation data (fallback)
        let diseaseKey = "plant_\(plant.id)_disease"
        let recommendationKey = "plant_\(plant.id)_recommendation"
        let processedPlantIdKey = "plant_\(plant.id)_processed_plant_id"
        
        // Load processed plant ID if available
        let cachedProcessedPlantId = UserDefaults.standard.integer(forKey: processedPlantIdKey)
        if cachedProcessedPlantId != 0 {
            processedPlantId = cachedProcessedPlantId
        }
        
        if let cachedDisease = UserDefaults.standard.string(forKey: diseaseKey),
           let cachedRecommendation = UserDefaults.standard.string(forKey: recommendationKey),
           !cachedDisease.isEmpty,
           !cachedRecommendation.isEmpty {
            // We have cached data - use it immediately
            estimatedDisease = cachedDisease
            recommendation = cachedRecommendation
            recommendationLoaded = true
            isLoadingRecommendation = false
            recommendationError = false
            userOpinion = ""
            print("[ViewModel] Loaded cached recommendation for plant \(plant.id)")
            return
        }
        
        // No cached data - start loading from WebSocket
        isLoadingRecommendation = true
        recommendationLoaded = false
        recommendationError = false
        estimatedDisease = ""
        recommendation = ""
        userOpinion = ""

        // Wire callback — must happen BEFORE any message could arrive
        // The callback now filters by plantId to avoid cross-plant contamination
        PlantWebSocketService.shared.onRecommendationReceived = {
            [weak self] plantId, processedPlantIdParam, disease, text in
            guard let self else { return }
            
            // Only process if this recommendation is for the current plant
            guard plantId == self.currentPlantId else {
                print("[ViewModel] Ignoring recommendation for plant \(plantId), waiting for \(self.currentPlantId ?? -1)")
                return
            }
            
            self.recommendationReceived(disease: disease, text: text, processedPlantId: processedPlantIdParam)
        }

        // 30 second timeout
        loadingTask?.cancel()
        loadingTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 30_000_000_000)
            guard !Task.isCancelled else { return }
            guard let self else { return }
            if !self.recommendationLoaded {
                self.isLoadingRecommendation = false
                self.recommendationError = true
                print("[ViewModel] Recommendation timed out")
            }
        }

        print("[ViewModel] Waiting for recommendation for plant \(plant.id)")
    }

    func recommendationReceived(disease: String, text: String, processedPlantId: Int?) {
        guard let plantId = currentPlantId else { return }
        
        loadingTask?.cancel()
        estimatedDisease = disease
        recommendation = text
        self.processedPlantId = processedPlantId
        
        // Persist recommendation data
        UserDefaults.standard.set(disease, forKey: "plant_\(plantId)_disease")
        UserDefaults.standard.set(text, forKey: "plant_\(plantId)_recommendation")
        if let processedPlantId = processedPlantId {
            UserDefaults.standard.set(processedPlantId, forKey: "plant_\(plantId)_processed_plant_id")
        }
        
        withAnimation(.spring(response: 0.5, dampingFraction: 0.8)) {
            recommendationLoaded = true
            isLoadingRecommendation = false
        }
        
        print("[ViewModel] Recommendation received and cached for plant \(plantId)")
    }

    func respond(accepted: Bool) {
        guard let plantId = currentPlantId else { return }
        hasResponded = true
        responseAccepted = accepted
        
        // Persist response state
        UserDefaults.standard.set(true, forKey: "plant_\(plantId)_responded")
        UserDefaults.standard.set(accepted, forKey: "plant_\(plantId)_accepted")
        
        // Only send accept immediately; reject will be sent with comment in submitOpinion()
        if accepted {
            Task {
                guard let processedPlantId = self.processedPlantId else {
                    print("[ViewModel] Cannot accept: processed_plant_id is missing")
                    return
                }
                
                do {
                    try await APIClient.shared.post(
                        endpoint: "/processed-plants/\(processedPlantId)/accept",
                        body: EmptyBody(),
                        requiresAuth: true
                    )
                    print("[ViewModel] Accepted for processedPlantId=\(processedPlantId)")
                } catch {
                    print("Failed to send accept:", error)
                }
            }
        }
    }

    func submitOpinion(plantId: Int) async {
        let trimmed = userOpinion.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return }
        
        guard let processedPlantId = self.processedPlantId else {
            print("Failed to submit opinion: processed_plant_id is missing")
            return
        }
        
        isSubmittingOpinion = true
        defer { isSubmittingOpinion = false }
        
        do {
            // First, send reject with comment
            try await APIClient.shared.post(
                endpoint: "/processed-plants/\(processedPlantId)/reject",
                body: RejectBody(comment: trimmed),
                requiresAuth: true
            )
            
            print("[ViewModel] Rejected with comment for processedPlantId=\(processedPlantId)")
            
            withAnimation { 
                opinionSubmitted = true 
                // Persist opinion submission state
                UserDefaults.standard.set(true, forKey: "plant_\(plantId)_opinion_submitted")
            }
            userOpinion = ""
        } catch {
            print("Failed to submit opinion:", error)
        }
    }

    func cancel() {
        loadingTask?.cancel()
        PlantWebSocketService.shared.onRecommendationReceived = nil
    }
}
