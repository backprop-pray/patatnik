package com.patatnik.server.service.plant.impl;

import com.patatnik.server.dto.PlantResponse;
import com.patatnik.server.exception.BadRequestException;
import com.patatnik.server.model.Plant;
import com.patatnik.server.model.ProcessedPlant;
import com.patatnik.server.model.User;
import com.patatnik.server.repository.PlantRepository;
import com.patatnik.server.repository.ProcessedPlantRepository;
import com.patatnik.server.service.CloudinaryService;
import com.patatnik.server.service.plant.PlantService;
import com.patatnik.server.service.recomendation.RecommendationService;

import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

import org.springframework.messaging.simp.SimpMessagingTemplate;

@Service
@RequiredArgsConstructor
public class PlantServiceImpl implements PlantService {

    private final PlantRepository plantRepository;
    private final CloudinaryService cloudinaryService;
    private final SimpMessagingTemplate messagingTemplate;
    private final RecommendationService recommendationService;

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

        PlantResponse response = PlantResponse.fromEntity(plant);

        messagingTemplate.convertAndSendToUser(
                user.getId().toString(),
                "/queue/plants",
                response
        );
        System.out.println("THE MESSAGE IS SEND");

        recommendationService.requestAnalysis(plant, user);

        return response;
    }

    @Override
    public java.util.List<PlantResponse> getPlantsByUser(User user) {
        return plantRepository
            .findByUserIdWithProcessedOrderByCreatedAtDesc(user.getId())
            .stream()
            .map(PlantResponse::fromEntity)
            .collect(java.util.stream.Collectors.toList());
    }

    private final ProcessedPlantRepository processedPlantRepository;

    @Override
    public void accept(Long id, User user) {
        ProcessedPlant pp = findAndValidate(id, user);
        pp.accept();
        processedPlantRepository.save(pp);
    }

    @Override
    public void reject(Long id, String comment, User user) {
        ProcessedPlant pp = findAndValidate(id, user);

        processedPlantRepository.updateRecommendedActionByDisease(
            pp.getDisease(),
            comment
        );

        pp.reject();
        processedPlantRepository.save(pp);
    }

    private ProcessedPlant findAndValidate(Long id, User user) {
        ProcessedPlant pp = processedPlantRepository.findById(id)
            .orElseThrow(() -> new BadRequestException("Processed plant not found"));

        if (!pp.getPlant().getUser().getId().equals(user.getId())) {
            throw new BadRequestException("Processed plant not found");
        }

        return pp;
    }

}
