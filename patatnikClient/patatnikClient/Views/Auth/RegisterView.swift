import SwiftUI

struct RegisterView: View {
    @EnvironmentObject var authVM: AuthViewModel
    @Environment(\.dismiss) private var dismiss

    @State private var fullName = ""
    @State private var email = ""
    @State private var password = ""
    @State private var confirmPassword = ""
    @State private var appeared = false
    @State private var fieldErrors: [Field: String] = [:]

    private enum Field: Hashable {
        case name, email, password, confirm
    }

    private var isFormValid: Bool {
        !fullName.isEmpty && !email.isEmpty && !password.isEmpty && !confirmPassword.isEmpty
    }

    var body: some View {
        ZStack {
            Color(hex: 0x0A0A0F).ignoresSafeArea()

            VStack(spacing: 0) {
                // Robot illustration + branding (compact for register)
                VStack(spacing: 4) {
                    RobotIllustration()
                        .scaleEffect(0.75)
                        .frame(height: 150)

                    Text("TankPlant")
                        .font(.title3.bold())
                        .foregroundStyle(.white)

                    Text("Autonomous Plant Intelligence")
                        .font(.caption).italic()
                        .foregroundStyle(Color(hex: 0x8E8E93))
                }
                .frame(height: 200)
                .staggerIn(index: 0, appeared: appeared)

                Spacer().frame(height: 16)

                // Form card
                ScrollView {
                    VStack(spacing: 20) {
                        // Error banner
                        if let error = authVM.errorMessage {
                            errorBanner(error)
                                .staggerIn(index: 1, appeared: appeared)
                        }

                        UnderlineField(
                            label: "FULL NAME",
                            text: $fullName,
                            errorMessage: fieldErrors[.name],
                            contentType: .name
                        )
                        .staggerIn(index: 2, appeared: appeared)
                        .onChange(of: fullName) { fieldErrors[.name] = nil }

                        UnderlineField(
                            label: "EMAIL",
                            text: $email,
                            errorMessage: fieldErrors[.email],
                            contentType: .emailAddress,
                            keyboardType: .emailAddress
                        )
                        .staggerIn(index: 3, appeared: appeared)
                        .onChange(of: email) { fieldErrors[.email] = nil }

                        UnderlineField(
                            label: "PASSWORD",
                            text: $password,
                            isSecure: true,
                            errorMessage: fieldErrors[.password],
                            contentType: .newPassword
                        )
                        .staggerIn(index: 4, appeared: appeared)
                        .onChange(of: password) { fieldErrors[.password] = nil }

                        UnderlineField(
                            label: "CONFIRM PASSWORD",
                            text: $confirmPassword,
                            isSecure: true,
                            errorMessage: fieldErrors[.confirm],
                            contentType: .newPassword
                        )
                        .staggerIn(index: 5, appeared: appeared)
                        .onChange(of: confirmPassword) { fieldErrors[.confirm] = nil }

                        // Create Account button
                        Button {
                            if validate() {
                                Task {
                                    await authVM.register(
                                        name: fullName,
                                        email: email,
                                        password: password
                                    )
                                }
                            }
                        } label: {
                            Group {
                                if authVM.isLoading {
                                    ProgressView()
                                        .tint(.white)
                                } else {
                                    Text("Create Account")
                                        .font(.headline)
                                }
                            }
                            .foregroundStyle(.white)
                            .frame(maxWidth: .infinity)
                            .frame(height: 54)
                            .background(
                                LinearGradient(
                                    colors: [Color(hex: 0xFF9500), Color(hex: 0xFF6B00)],
                                    startPoint: .leading,
                                    endPoint: .trailing
                                )
                            )
                            .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                            .shadow(color: Color(hex: 0xFF9500).opacity(0.35), radius: 12, y: 4)
                        }
                        .buttonStyle(ScaleButtonStyle())
                        .disabled(!isFormValid || authVM.isLoading)
                        .opacity(isFormValid && !authVM.isLoading ? 1 : 0.6)
                        .staggerIn(index: 6, appeared: appeared)

                        // Sign In link
                        HStack(spacing: 4) {
                            Text("Already have an account?")
                                .font(.subheadline)
                                .foregroundStyle(Color(hex: 0x8E8E93))

                            Button {
                                dismiss()
                            } label: {
                                Text("Sign In")
                                    .font(.subheadline.bold())
                                    .foregroundStyle(Color(hex: 0xFF9500))
                            }
                        }
                        .staggerIn(index: 7, appeared: appeared)
                    }
                    .padding(.horizontal, 28)
                    .padding(.vertical, 32)
                }
                .background(
                    RoundedRectangle(cornerRadius: 28, style: .continuous)
                        .fill(Color(hex: 0x1C1C1E))
                        .shadow(color: .black.opacity(0.4), radius: 24, y: -8)
                )
            }
        }
        .navigationBarBackButtonHidden()
        .onAppear {
            withAnimation { appeared = true }
            authVM.errorMessage = nil
        }
    }

    // MARK: - Validation

    private func validate() -> Bool {
        var errors: [Field: String] = [:]

        if fullName.trimmingCharacters(in: .whitespaces).isEmpty {
            errors[.name] = "Name is required"
        }

        let trimmedEmail = email.trimmingCharacters(in: .whitespaces)
        if trimmedEmail.isEmpty {
            errors[.email] = "Email is required"
        } else if !trimmedEmail.contains("@") || !trimmedEmail.contains(".") {
            errors[.email] = "Enter a valid email address"
        }

        if password.count < 8 {
            errors[.password] = "Password must be at least 8 characters"
        }

        if confirmPassword != password {
            errors[.confirm] = "Passwords do not match"
        }

        withAnimation(.easeInOut(duration: 0.2)) {
            fieldErrors = errors
        }

        return errors.isEmpty
    }

    // MARK: - Error Banner

    private func errorBanner(_ message: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.subheadline)
            Text(message)
                .font(.subheadline)
        }
        .foregroundStyle(Color(hex: 0xFF453A))
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Color(hex: 0xFF453A).opacity(0.12))
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .stroke(Color(hex: 0xFF453A).opacity(0.3), lineWidth: 1)
                )
        )
        .transition(.move(edge: .top).combined(with: .opacity))
    }
}

#Preview {
    NavigationStack {
        RegisterView()
            .environmentObject(AuthViewModel())
    }
}
