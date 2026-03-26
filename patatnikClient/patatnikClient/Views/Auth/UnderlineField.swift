import SwiftUI

struct UnderlineField: View {
    let label: String
    @Binding var text: String
    var isSecure: Bool = false
    var errorMessage: String? = nil
    var contentType: UITextContentType? = nil
    var keyboardType: UIKeyboardType = .default

    @FocusState private var isFocused: Bool
    @State private var showSecureText = false

    private var hasError: Bool { errorMessage != nil }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(label)
                .font(.caption)
                .foregroundStyle(Color(hex: 0x8E8E93))

            HStack(spacing: 8) {
                Group {
                    if isSecure && !showSecureText {
                        SecureField("", text: $text)
                            .textContentType(contentType)
                    } else {
                        TextField("", text: $text)
                            .textContentType(contentType)
                            .keyboardType(keyboardType)
                    }
                }
                .font(.body)
                .foregroundStyle(.white)
                .focused($isFocused)
                .autocorrectionDisabled()
                .textInputAutocapitalization(.never)

                if isSecure {
                    Button {
                        showSecureText.toggle()
                    } label: {
                        Image(systemName: showSecureText ? "eye.slash.fill" : "eye.fill")
                            .font(.subheadline)
                            .foregroundStyle(Color(hex: 0x8E8E93))
                    }
                }
            }

            Rectangle()
                .fill(hasError
                      ? Color(hex: 0xFF453A)
                      : isFocused
                        ? Color(hex: 0xFF9500)
                        : Color.white.opacity(0.15))
                .frame(height: 1)
                .animation(.easeInOut(duration: 0.2), value: isFocused)
                .animation(.easeInOut(duration: 0.2), value: hasError)

            if let error = errorMessage {
                Text(error)
                    .font(.caption2)
                    .foregroundStyle(Color(hex: 0xFF453A))
                    .transition(.opacity)
            }
        }
    }
}

#Preview {
    ZStack {
        Color(hex: 0x0A0A0F).ignoresSafeArea()
        VStack(spacing: 24) {
            UnderlineField(label: "EMAIL", text: .constant("test@test.com"))
            UnderlineField(label: "PASSWORD", text: .constant(""), isSecure: true)
            UnderlineField(label: "ERROR", text: .constant(""), errorMessage: "This field is required")
        }
        .padding(28)
    }
}
