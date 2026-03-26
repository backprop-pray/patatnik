import Foundation

struct Plant: Codable, Identifiable {
    let id: Int
    let latitude: Double
    let longitude: Double
    let imageUrl: String?
    let userId: Int?
    let createdAt: String?
}
