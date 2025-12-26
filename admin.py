from starlette_admin.contrib.sqla import Admin, ModelView
from database import engine
from models import User, Survey, Caption, Response, AgentEvalDetail, AgentEvalDetailUnseen

class UserAdmin(ModelView):
    fields = ['userId', 'username', 'gender', 'email'] # hide password
    searchable_fields = ['username', 'email', 'gender']
    sortable_fields = ['userId', 'username', 'email', 'gender']
    export_columns = ['userId', 'username', 'gender', 'email']

class SurveyAdmin(ModelView):
    fields = ['surveyId', 'userId', 'user', 'title', 'category', 'country', 'imageUrl']
    searchable_fields = ['title', 'category', 'country', 'user.username', 'user.email']
    sortable_fields = ['surveyId', 'title', 'category', 'country', 'userId']
    export_columns = ['surveyId', 'title', 'category', 'country', 'user.username', 'imageUrl']

class CaptionAdmin(ModelView):
    fields = ['captionId', 'surveyId', 'survey', 'text', 'type']
    searchable_fields = ['text', 'type', 'survey.title', 'survey.country']
    sortable_fields = ['captionId', 'surveyId', 'type']
    export_columns = ['captionId', 'survey.title', 'text', 'type']

class ResponseAdmin(ModelView):
    fields = ['responseId', 'userId', 'user', 'captionId', 'caption', 'cultural', 'visual', 'hallucination', 'time', 'created_at']
    searchable_fields = ['user.username', 'user.email', 'caption.text', 'caption.type']
    sortable_fields = ['responseId', 'userId', 'captionId', 'cultural', 'visual', 'hallucination', 'time', 'created_at']
    export_columns = ['responseId', 'user.username', 'caption.text', 'cultural', 'visual', 'hallucination', 'time', 'created_at']

class AgentEvaluationAdmin(ModelView):
    fields = ['Id', 'captionId', 'caption', 'cutural', 'visual', 'hallucination']

class AgentEvalDetailAdmin(ModelView):
    fields = ['id', 'type', 'likert', 'value', 'flag', 'captionId', 'caption']
    searchable_fields = ['type', 'caption.text', 'caption.type', 'caption.survey.title']
    sortable_fields = ['id', 'type', 'likert', 'value', 'flag', 'captionId']
    export_columns = ['id', 'type', 'likert', 'value', 'flag', 'caption.text']

class AgentEvalDetailUnseenAdmin(ModelView):
    fields = ['id', 'type', 'likert', 'value', 'flag', 'captionId', 'caption']
    searchable_fields = ['type', 'caption.text', 'caption.type', 'caption.survey.title']
    sortable_fields = ['id', 'type', 'likert', 'value', 'flag', 'captionId']
    export_columns = ['id', 'type', 'likert', 'value', 'flag', 'caption.text']

# class LearningDataView(ModelView):
#     name = "Learning Data"
#     identity = "learning-data"
#     fields = [
#         'responseId',
#         'user',
#         'caption.text',
#         'survey.title',
#         'cultural',
#         'visual',
#         'hallucination',
#     ]

admin_app = Admin(engine, title="CultureLens Admin", templates_dir="templates")

admin_app.add_view(UserAdmin(User, icon="fa fa-users"))
admin_app.add_view(SurveyAdmin(Survey, icon="fa fa-list-alt"))
admin_app.add_view(CaptionAdmin(Caption, icon="fa fa-closed-captioning"))
admin_app.add_view(ResponseAdmin(Response, icon="fa fa-comment"))
admin_app.add_view(AgentEvalDetailAdmin(AgentEvalDetail, icon="fa fa-info-circle"))
admin_app.add_view(AgentEvalDetailUnseenAdmin(AgentEvalDetailUnseen, icon="fa fa-eye-slash"))
# admin_app.add_view(LearningDataView(Response, icon="fa fa-flask"))
