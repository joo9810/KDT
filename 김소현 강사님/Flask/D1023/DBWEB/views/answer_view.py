# --------------------------------------------------------------------------------
# Flask Framework에서 모듈단위 URL 처리 파일
# - 파일명 : answer_view.py
# --------------------------------------------------------------------------------
# 모듈 로딩 ----------------------------------------------------------------------
from flask import Blueprint, url_for, request
from datetime import datetime
from werkzeug.utils import redirect
from DBWEB import DB
from DBWEB.models.models import Question, Answer

# Blueprint 인스턴스 생성
answerBP = Blueprint('ANSWER',
                     import_name=__name__,
                     url_prefix='/answer',
                     template_folder='templates')

@answerBP.route('/create/<int:question_id>', methods=('POST', ))
def create(question_id):
    question = Question.query.get_or_404(question_id)
    content = request.form['content'] # name 속성이 'content'인 값
    answer = Answer(content=content, create_date=datetime.now())
    question.answer_set.append(answer)
    DB.session.commit()
    return redirect(f'/qdetail/{question_id}')
