-- Check current table structure
DESCRIBE `agentEvaluation`;

-- Show any existing data
SELECT * FROM `agentEvaluation` LIMIT 5;

-- Fix the table by altering it to have AUTO_INCREMENT
-- First clear any existing data
DELETE FROM `agentEvaluation`;

-- Drop the table and recreate it properly
DROP TABLE IF EXISTS `agentEvaluation`;

CREATE TABLE `agentEvaluation` (
    `Id` BIGINT NOT NULL AUTO_INCREMENT,
    `cutural` FLOAT NULL,
    `visual` FLOAT NULL,
    `hallucination` FLOAT NULL,
    `captionId` BIGINT NOT NULL,
    PRIMARY KEY (`Id`),
    INDEX `idx_captionId` (`captionId`),
    CONSTRAINT `fk_agentEvaluation_caption`
        FOREIGN KEY (`captionId`) REFERENCES `caption` (`captionId`)
        ON DELETE CASCADE
);

-- Verify the new structure
DESCRIBE `agentEvaluation`;
SHOW CREATE TABLE `agentEvaluation`;