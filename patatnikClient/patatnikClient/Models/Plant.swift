import Foundation

nonisolated struct Plant: Codable, Identifiable, Sendable {
    let id: Int
    let latitude: Double
    let longitude: Double
    let imageUrl: String?
    let userId: Int?
    let createdAt: String?
}
