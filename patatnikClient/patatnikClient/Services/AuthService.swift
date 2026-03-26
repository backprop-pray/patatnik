import Foundation

struct AuthService {

    enum AuthError: Error {
        case invalidURL
        case invalidCredentials
        case serverError(String)
        case networkError
        case decodingError(String)

        var userMessage: String {
            switch self {
            case .invalidURL:
                return "Configuration error. Please contact support."
            case .invalidCredentials:
                return "Invalid email or password."
            case .serverError(let message):
                return message
            case .networkError:
                return "Network error. Check your connection."
            case .decodingError(let details):
                return "App out of date or server changed. Details: \(details)"
            }
        }
    }

    func login(email: String, password: String) async throws -> AuthResponse {
        guard let url = URL(string: "\(AppConfig.baseURL)/auth/login") else {
            throw AuthError.invalidURL
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(LoginBody(email: email, password: password))

        let authResponse = try await perform(request)
        await APIClient.shared.setToken(authResponse.token)
        return authResponse
    }

    func register(name: String, email: String, password: String) async throws -> AuthResponse {
        guard let url = URL(string: "\(AppConfig.baseURL)/auth/register") else {
            throw AuthError.invalidURL
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(
            RegisterBody(username: name, email: email, password: password)
        )

        let authResponse = try await perform(request)
        await APIClient.shared.setToken(authResponse.token)
        return authResponse
    }

    // MARK: - Private

    private func perform(_ request: URLRequest) async throws -> AuthResponse {
        let (data, response): (Data, URLResponse)
        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch {
            throw AuthError.networkError
        }

        guard let http = response as? HTTPURLResponse else {
            throw AuthError.networkError
        }

        if http.statusCode == 401 {
            throw AuthError.invalidCredentials
        }

        guard (200...299).contains(http.statusCode) else {
            if let body = try? JSONDecoder().decode(ServerErrorBody.self, from: data) {
                throw AuthError.serverError(body.message)
            }
            throw AuthError.serverError("Server error (\(http.statusCode))")
        }

        do {
            let envelope = try JSONDecoder().decode(APIEnvelope<AuthResponse>.self, from: data)
            guard let authData = envelope.data else {
                throw AuthError.decodingError("Data was null on success")
            }
            return authData
        } catch let error as AuthError {
            throw error
        } catch {
            throw AuthError.decodingError(error.localizedDescription)
        }
    }
}

// MARK: - Request Bodies

private struct LoginBody: Encodable {
    let email: String
    let password: String
}

private struct RegisterBody: Encodable {
    let username: String
    let email: String
    let password: String
}

private struct ServerErrorBody: Decodable {
    let message: String
}
