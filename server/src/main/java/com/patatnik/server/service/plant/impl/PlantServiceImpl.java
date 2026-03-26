package com.patatnik.server.service.plant.impl;

import com.patatnik.server.dto.PlantResponse;
import com.patatnik.server.exception.BadRequestException;
import com.patatnik.server.model.Plant;
import com.patatnik.server.model.User;
import com.patatnik.server.repository.PlantRepository;
import com.patatnik.server.service.CloudinaryService;
import com.patatnik.server.service.plant.PlantService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@Service
@RequiredArgsConstructor
public class PlantServiceImpl implements PlantService {

    private final PlantRepository plantRepository;
    private final CloudinaryService cloudinaryService;

    @Override
    public PlantResponse createPlant(Double latitude, Double longitude, MultipartFile image, User user) {
        String imageUrl = null;

        if (image != null && !image.isEmpty()) {
            try {
                imageUrl = cloudinaryService.uploadImage(image);
            } catch (IOException e) {
                throw new BadRequestException("Failed to upload image to Cloudinary");
            }
        }

        Plant plant = Plant.builder()
                .latitude(latitude)
                .longitude(longitude)
                .imageUrl(imageUrl)
                .user(user)
                .build();

        plant = plantRepository.save(plant);

        return PlantResponse.fromEntity(plant);
    }
}
