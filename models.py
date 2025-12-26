
from sqlalchemy import Column, BigInteger, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from database import Base

class User(Base):
    __tablename__ = "user"

    userId = Column(BigInteger, primary_key=True, index=True)
    username = Column(String(255), unique=True, index=True)
    password = Column(String(255))
    gender = Column(String(255))
    email = Column(String(255), unique=True)
    surveys = relationship("Survey", back_populates="user")
    responses = relationship("Response", back_populates="user")

    def __admin_repr__(self, request):
        return f"{self.username} ({self.email})"

class Survey(Base):
    __tablename__ = "survey"
    surveyId = Column(BigInteger, primary_key=True, autoincrement=True)
    imageUrl = Column(String(255))
    country = Column(String(255))
    category = Column(String(255))
    title = Column(String(255))
    userId = Column(BigInteger, ForeignKey("user.userId"))
    user = relationship("User", back_populates="surveys")
    captions = relationship("Caption", back_populates="survey")

    def __admin_repr__(self, request):
        return f"{self.title} ({self.country})"

class Caption(Base):
    __tablename__ = "caption"

    captionId = Column(BigInteger, primary_key=True, index=True)
    surveyId = Column(BigInteger, ForeignKey("survey.surveyId"))
    text = Column(String(255))
    type = Column(String(255))

    survey = relationship("Survey", back_populates="captions")
    responses = relationship("Response", back_populates="caption")

    def __admin_repr__(self, request):
        return f"{self.text[:50]}... ({self.type})" if len(self.text) > 50 else f"{self.text} ({self.type})"

class Response(Base):
    __tablename__ = "response"

    responseId = Column(BigInteger, primary_key=True, index=True)
    cultural = Column(Integer)
    visual = Column(Integer)
    userId = Column(BigInteger, ForeignKey("user.userId"))
    captionId = Column(BigInteger, ForeignKey("caption.captionId"))
    hallucination = Column(Integer)
    time = Column(Float, default=0)
    created_at = Column(DateTime)

    user = relationship("User", back_populates="responses")
    caption = relationship("Caption", back_populates="responses")

    @property
    def caption_text(self):
        return self.caption.text

    @property
    def survey_title(self):
        return self.caption.survey.title

class AgentEvalDetail(Base):
    __tablename__ = "agentEvalDetail"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    type = Column(String(255), nullable=False, comment='Type of evaluation (cultural, visual, hallucination)')
    likert = Column(Integer, nullable=False, comment='Likert scale value (1-5)')
    value = Column(Float, nullable=False, comment='Numeric value from distribution')
    flag = Column(Integer, nullable=False, comment='Flag value from API request')
    captionId = Column(BigInteger, ForeignKey("caption.captionId"), nullable=False)

    caption = relationship("Caption")

class AgentEvalDetailV2(Base):
    __tablename__ = "agentEvalDetail_v2"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    type = Column(String(255), nullable=False, comment='Type of evaluation (cultural, visual, hallucination)')
    likert = Column(Integer, nullable=False, comment='Likert scale value (1-5)')
    value = Column(Float, nullable=False, comment='Numeric value from distribution')
    flag = Column(Integer, nullable=False, comment='Flag value from API request')
    captionId = Column(BigInteger, ForeignKey("caption.captionId"), nullable=False)

    caption = relationship("Caption")

class AgentEvalDetailUnseen(Base):
    __tablename__ = "agentEvalDetail_unseen"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    type = Column(String(255), nullable=False, comment='Type of evaluation (cultural, visual, hallucination)')
    likert = Column(Integer, nullable=False, comment='Likert scale value (1-5)')
    value = Column(Float, nullable=False, comment='Numeric value from distribution')
    flag = Column(Integer, nullable=False, comment='Flag value from API request')
    captionId = Column(BigInteger, ForeignKey("caption.captionId"), nullable=False)

    caption = relationship("Caption")

    def __admin_repr__(self, request):
        return f"{self.type} - Likert:{self.likert} (Flag:{self.flag})"