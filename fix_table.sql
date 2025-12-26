-- Fix the agentEvaluation table to have proper AUTO_INCREMENT
-- First, drop and recreate the table with correct structure

DROP TABLE IF EXISTS `agentEvaluation`;

CREATE TABLE `agentEvaluation` (
    `Id` BIGINT NOT NULL AUTO_INCREMENT,
    `cutural` FLOAT NULL,
    `visual` FLOAT NULL,
    `hallucination` FLOAT NULL,
    `captionId` BIGINT NOT NULL,
    PRIMARY KEY (`Id`),
    INDEX `idx_captionId` (`captionId`),
    FOREIGN KEY (`captionId`) REFERENCES `caption` (`captionId`)
);