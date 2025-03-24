from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

JWT_SECRET = "your-secret-key"
JWT_ALGO = "HS256"
security = HTTPBearer()

def verify_token(token: str):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def require_jwt(credentials: HTTPAuthorizationCredentials = Depends(security)):
    return verify_token(credentials.credentials)
