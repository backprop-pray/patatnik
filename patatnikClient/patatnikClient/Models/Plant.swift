import Foundation

nonisolated struct Plant: Codable, Identifiable, Sendable {
    let id: Int
    let latitude: Double
    let longitude: Double
    let imageUrl: String?
    let userId: Int?
    let createdAt: String?
    let disease: String?
    let recommendedAction: String?
    let status: Bool?
    let processedPlantId: Int?
    
    enum CodingKeys: String, CodingKey {
        case id
        case latitude
        case longitude
        case imageUrl
        case userId
        case createdAt
        case disease
        case recommendedAction
        case status
        case processedPlantId = "processPlantId"  // Backend sends "processPlantId" without "ed"
    }
    
    // Custom decoder - now simpler since we know the exact field name
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(Int.self, forKey: .id)
        latitude = try container.decode(Double.self, forKey: .latitude)
        longitude = try container.decode(Double.self, forKey: .longitude)
        imageUrl = try container.decodeIfPresent(String.self, forKey: .imageUrl)
        userId = try container.decodeIfPresent(Int.self, forKey: .userId)
        createdAt = try container.decodeIfPresent(String.self, forKey: .createdAt)
        disease = try container.decodeIfPresent(String.self, forKey: .disease)
        recommendedAction = try container.decodeIfPresent(String.self, forKey: .recommendedAction)
        status = try container.decodeIfPresent(Bool.self, forKey: .status)
        processedPlantId = try container.decodeIfPresent(Int.self, forKey: .processedPlantId)
    }
    
    // Manual init for previews/tests
    init(id: Int, latitude: Double, longitude: Double, imageUrl: String?, userId: Int?, createdAt: String?, disease: String?, recommendedAction: String?, status: Bool?, processedPlantId: Int?) {
        self.id = id
        self.latitude = latitude
        self.longitude = longitude
        self.imageUrl = imageUrl
        self.userId = userId
        self.createdAt = createdAt
        self.disease = disease
        self.recommendedAction = recommendedAction
        self.status = status
        self.processedPlantId = processedPlantId
    }

    /// Returns the image URL with HTTP upgraded to HTTPS to satisfy ATS policy.
    var safeImageUrl: URL? {
        guard let imageUrl, !imageUrl.isEmpty else { return nil }
        let secured = imageUrl.hasPrefix("http://")
            ? imageUrl.replacingOccurrences(of: "http://", with: "https://")
            : imageUrl
        return URL(string: secured)
    }
    
    /// Checks if a recommendation exists (from backend or cache)
    var hasRecommendation: Bool {
        // First check if the plant object itself has recommendation data
        if let disease = disease, let recommendedAction = recommendedAction,
           !disease.isEmpty, !recommendedAction.isEmpty {
            return true
        }
        
        // Fallback to local cache
        let diseaseKey = "plant_\(id)_disease"
        let recommendationKey = "plant_\(id)_recommendation"
        
        guard let cachedDisease = UserDefaults.standard.string(forKey: diseaseKey),
              let cachedRecommendation = UserDefaults.standard.string(forKey: recommendationKey) else {
            return false
        }
        
        return !cachedDisease.isEmpty && !cachedRecommendation.isEmpty
    }
}
