package com.patatnik.server.service.plant;

import com.patatnik.server.dto.PlantResponse;
import com.patatnik.server.model.User;
import org.springframework.web.multipart.MultipartFile;

public interface PlantService {
    PlantResponse createPlant(Double latitude, Double longitude, MultipartFile image, User user);

    java.util.List<PlantResponse> getPlantsByUser(User user);
}
