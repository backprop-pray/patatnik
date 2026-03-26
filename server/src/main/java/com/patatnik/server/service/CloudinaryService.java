package com.patatnik.server.service;

import com.cloudinary.Cloudinary;
import com.cloudinary.utils.ObjectUtils;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.Map;

@Service
@RequiredArgsConstructor
public class CloudinaryService {

    private final Cloudinary cloudinary;

    @SuppressWarnings("unchecked")
    public String uploadImage(MultipartFile file) throws IOException {
        Map<String, Object> options = ObjectUtils.asMap("folder", "Plants");
        Map<String, Object> uploadResult = cloudinary.uploader().upload(file.getBytes(), options);
        return uploadResult.get("url").toString();
    }
}
