-- 기존 테이블 삭제하고 올바른 AUTO_INCREMENT로 다시 생성

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
) ENGINE=InnoDB;