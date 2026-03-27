package com.patatnik.server.service.processedPlant.impl;

import com.patatnik.server.exception.BadRequestException;
import com.patatnik.server.model.ProcessedPlant;
import com.patatnik.server.model.User;
import com.patatnik.server.repository.ProcessedPlantRepository;
import com.patatnik.server.service.processedPlant.ProcessedPlantService;

import jakarta.transaction.Transactional;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
@Transactional
public class ProcessedPlantServiceImpl implements ProcessedPlantService {

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