from datetime import datetime, timedelta, timezone
from pathlib import Path
import os, secrets
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

from app import database as db, evaluator as ev, recommender as rec, resume as rs
from app.config import ENV_API_KEY, JOB_ROLES, CATEGORIES, DIFFICULTIES
from app.llm import LLMClient

SECRET = os.getenv('APP_SECRET', 'change-this-secret-before-production')
serializer = URLSafeTimedSerializer(SECRET, salt='ai-interview-prep-auth')
app = FastAPI(title='AI Interview Prep API', version='1.0.0')
origins = [x.strip() for x in os.getenv('FRONTEND_ORIGINS','http://localhost:5173').split(',') if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
db.init_db()

class Credentials(BaseModel): username: str = Field(min_length=3, max_length=50); password: str = Field(min_length=6, max_length=128)
class Profile(BaseModel):
    full_name: str = ''; email: str = ''; branch: str = ''; college: str = ''; grad_year: int = 2027
    target_role: str = 'Software Developer'; level: str = 'Fresher'
class InterviewStart(BaseModel):
    role: str = 'Software Developer'; category: str = 'Mixed'; difficulty: str = 'Intermediate'; count: int = Field(default=5, ge=3, le=10)
class AnswerBody(BaseModel): question_index: int = Field(ge=0); question: dict; answer: str = Field(min_length=1, max_length=12000)
class FinishBody(BaseModel): items: list[dict]

sessions = {}
def token_for(uid): return serializer.dumps({'uid': uid})
def current_user(authorization: Optional[str] = Header(default=None)):
    if not authorization or not authorization.startswith('Bearer '): raise HTTPException(401, 'Please log in.')
    try: data = serializer.loads(authorization[7:], max_age=60*60*24*7)
    except (BadSignature, SignatureExpired): raise HTTPException(401, 'Session expired. Please log in again.')
    return int(data['uid'])
def llm(): return LLMClient(api_key=ENV_API_KEY)

@app.get('/')
def root(): return {'message':'AI Interview Prep API is running'}
@app.get('/health')
def health(): return {'status':'ok'}
@app.get('/api/meta')
def meta(): return {'roles':JOB_ROLES,'categories':CATEGORIES,'difficulties':DIFFICULTIES,'groq_configured':bool(ENV_API_KEY)}
@app.post('/api/auth/register')
def register(body: Credentials):
    ok,msg=db.register(body.username, body.password)
    if not ok: raise HTTPException(400,msg)
    uid=db.login(body.username,body.password)
    return {'token':token_for(uid),'username':body.username.strip().lower()}
@app.post('/api/auth/login')
def login(body: Credentials):
    uid=db.login(body.username,body.password)
    if not uid: raise HTTPException(401,'Invalid username or password.')
    return {'token':token_for(uid),'username':body.username.strip().lower()}
@app.get('/api/profile')
def get_profile(uid:int=Depends(current_user)): return db.get_profile(uid)
@app.put('/api/profile')
def put_profile(body:Profile, uid:int=Depends(current_user)):
    if body.target_role not in JOB_ROLES: raise HTTPException(400,'Invalid target role')
    db.save_profile(uid, **body.model_dump()); return {'message':'Profile saved'}
@app.post('/api/resume')
async def upload_resume(file: UploadFile=File(...), uid:int=Depends(current_user)):
    if not file.filename.lower().endswith('.pdf'): raise HTTPException(400,'Upload a PDF file.')
    data=await file.read()
    if len(data)>8*1024*1024: raise HTTPException(413,'PDF must be 8 MB or smaller.')
    try: text=rs.extract_pdf_text(data)
    except Exception as e: raise HTTPException(400,str(e))
    analysis=rs.analyse_resume(text,llm()); db.save_resume(uid,text,analysis)
    return {'analysis':analysis}
@app.get('/api/resume')
def get_resume(uid:int=Depends(current_user)):
    r=db.get_resume(uid)
    return {'analysis':r['analysis']} if r else {'analysis':None}
@app.post('/api/interviews/start')
def start_interview(body:InterviewStart, uid:int=Depends(current_user)):
    if body.role not in JOB_ROLES or body.category not in CATEGORIES or body.difficulty not in DIFFICULTIES: raise HTTPException(400,'Invalid interview settings')
    profile=db.get_profile(uid); resume=db.get_resume(uid); weak=rec.weak_topics_from_history(db.all_user_answers(uid))
    qs=ev.generate_questions(llm(),body.role,body.category,body.difficulty,body.count,profile,resume['analysis'] if resume else None,weak)
    iid=db.create_interview(uid,body.role,body.category,body.difficulty)
    sessions[iid]={'uid':uid,'role':body.role,'questions':qs,'items':[]}
    return {'interview_id':iid,'questions':qs,'demo_mode':llm().demo}
@app.post('/api/interviews/{iid}/answer')
def answer(iid:int, body:AnswerBody, uid:int=Depends(current_user)):
    state=sessions.get(iid)
    if not state or state['uid']!=uid: raise HTTPException(404,'Interview session not found. Start a new interview.')
    if body.question_index>=len(state['questions']): raise HTTPException(400,'Invalid question index')
    q=state['questions'][body.question_index]
    if q['question'] != body.question.get('question'): raise HTTPException(400,'Question does not match this interview')
    result=ev.evaluate_answer(llm(),q,body.answer)
    db.save_answer(iid,body.question_index,q,body.answer,result)
    item={'question':q['question'],'topic':q.get('topic','General'),'type':q.get('type','Technical'),'score':result['overall'],'eval':result,'answer':body.answer}
    state['items']=[x for x in state['items'] if x['question']!=q['question']]+[item]
    return {'evaluation':result}
@app.post('/api/interviews/{iid}/finish')
def finish(iid:int, uid:int=Depends(current_user)):
    state=sessions.get(iid)
    if not state or state['uid']!=uid: raise HTTPException(404,'Interview session not found.')
    items=state['items']
    if not items: raise HTTPException(400,'Submit at least one answer before finishing.')
    report=rec.build_report(llm(),state['role'],items)
    db.finish_interview(iid,report['overall'],report)
    sessions.pop(iid,None)
    return {'report':report,'items':items}
@app.get('/api/dashboard')
def dashboard(uid:int=Depends(current_user)):
    interviews=db.list_interviews(uid); answers=db.all_user_answers(uid)
    scores=[float(x['overall_score'] or 0) for x in interviews]
    by_topic={}
    for row in answers: by_topic.setdefault(row['topic'] or 'General',[]).append(float(row['score']))
    return {'interviews_count':len(interviews),'average_score':round(sum(scores)/len(scores),1) if scores else 0,'best_score':max(scores) if scores else 0,'interviews':interviews,'topic_scores':{k:round(sum(v)/len(v),1) for k,v in by_topic.items()},'weak_topics':rec.weak_topics_from_history(answers)}
@app.get('/api/history')
def history(uid:int=Depends(current_user)): return db.list_interviews(uid)
@app.get('/api/history/{iid}')
def history_detail(iid:int,uid:int=Depends(current_user)):
    d=db.get_interview(iid)
    if not d or d['user_id']!=uid: raise HTTPException(404,'Interview not found')
    return d
