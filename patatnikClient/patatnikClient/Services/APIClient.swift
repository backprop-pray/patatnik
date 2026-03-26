import Foundation

// MARK: - Shared API Types

nonisolated struct APIEnvelope<T: Decodable & Sendable>: Decodable, Sendable {
    let status: Int
    let message: String
    let data: T?
}

nonisolated enum APIError: Error, Sendable {
    case invalidURL
    case networkError
    case serverError(Int, String)
    case noData
    case decodingError(String)

    var errorDescription: String {
        switch self {
        case .invalidURL:
            return "Configuration error."
        case .networkError:
            return "Network error. Check your connection."
        case .serverError(let code, let message):
            return message.isEmpty ? "Server error (\(code))" : message
        case .noData:
            return "No data returned from server."
        case .decodingError(let details):
            return "Decoding error: \(details)"
        }
    }
}

// MARK: - API Client

actor APIClient {
    static let shared = APIClient()

    private var token: String?

    private init() {}

    func setToken(_ token: String?) {
        self.token = token
    }

    func get<T: Decodable & Sendable>(endpoint: String, requiresAuth: Bool = false) async throws -> APIEnvelope<T> {
        let baseURL = AppConfig.baseURL
        guard let url = URL(string: "\(baseURL)\(endpoint)") else {
            throw APIError.invalidURL
        }

        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        if requiresAuth {
            guard let token else {
                throw APIError.serverError(401, "Not authenticated.")
            }
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        let (data, response): (Data, URLResponse)
        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch {
            throw APIError.networkError
        }

        guard let http = response as? HTTPURLResponse else {
            throw APIError.networkError
        }

        guard (200...299).contains(http.statusCode) else {
            if let body = try? JSONDecoder().decode(APIErrorBody.self, from: data) {
                throw APIError.serverError(http.statusCode, body.message)
            }
            throw APIError.serverError(http.statusCode, "")
        }

        do {
            return try JSONDecoder().decode(APIEnvelope<T>.self, from: data)
        } catch {
            throw APIError.decodingError(error.localizedDescription)
        }
    }
}

nonisolated private struct APIErrorBody: Decodable, Sendable {
    let message: String
}
