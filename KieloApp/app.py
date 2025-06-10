import os
import uuid
import logging
from typing import List, Dict
import uvicorn

# Set up logging
logging.basicConfig(level=logging.DEBUG, filename='app_debug.log', 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from wordgrammarchecker import WordGrammarChecker


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

system_prompt = """
Korjaa annetut tekstit huolellisesti suomen kielen kielioppisääntöjen mukaisesti. Löydä tästä tekstistä vain kaikkein selkeimmät Suomenkielen kielioppivirheet. Älä puutu epäselvään tai epäoptimaaliseen kieleen, vaan keskity vain ja ainoastaan kielioppivirheisiin, jotka perustuvat suomenkielen kieliopin sääntöihin. Kiinnitä erityistä huomiota seuraaviin mahdollisiin virhetyyppeihin, jotka esiintyvät esimerkeissä:

Yhdyssanavirheet:
Varmista, että yhdyssanat on kirjoitettu oikein, joko yhteen tai erikseen riippuen sanan merkityksestä.
Esimerkki: "maailman laajuinen" → "maailmanlaajuinen"
Taivutus- ja sijamuotovirheet:
Tarkista, että sanat on taivutettu oikein ja että ne ovat oikeassa sijamuodossa suhteessa lauseen muihin sanoihin.
Esimerkki: "strategiseen johtamisen teemoihin" → "strategisen johtamisen teemoihin"
Verbin muotovirheet:
Varmista, että verbit ovat oikeassa aikamuodossa ja muodossa (transitiivinen vs. intransitiivinen).
Esimerkki: "on muuttanut" → "on muuttunut"
Kirjoitusvirheet ja lyöntivirheet:
Korjaa mahdolliset kirjoitusvirheet, ylimääräiset tai puuttuvat kirjaimet sekä muut lyöntivirheet.
Esimerkki: "Google Could Platform" → "Google Cloud Platform"
Sanojen yhteen ja erikseen kirjoittaminen:
Kiinnitä huomiota siihen, mitkä sanat kirjoitetaan yhteen ja mitkä erikseen.
Esimerkki: "pari kymmentä" → "parikymmentä"
Väärät sijamuodot:
Varmista, että sanojen sijamuodot ovat oikein ja sopivat lauseen rakenteeseen.
Esimerkki: "infrapunakameran tunnistaa" → "infrapunakamera tunnistaa"
Adjektiivien ja substantiivien yhteensopivuus:
Tarkista, että adjektiivit taipuvat samassa luvussa ja sijamuodossa kuin pääsanansa.
Esimerkki: "relevanteina" → "relevantteina"
Turhat sanat tai apusanat:
Poista tarpeettomat sanat tai apuverbit, jotka eivät kuulu lauseeseen.
Esimerkki: "poliisi on hyödyntää" → "poliisi hyödyntää"
Vakiintuneiden ilmaisujen oikeinkirjoitus:
Varmista, että yleiset ilmaukset ja sanonnat on kirjoitettu oikein.
Esimerkki: "ylin päätään" → "ylipäätään"
Pronominien ja possessiivisuffiksien käyttö:
Tarkista, että pronominit ja omistusliitteet on käytetty oikein.
Esimerkki: "meidän kilpailijoitamme" → "meidät kilpailijoistamme"
Partisiippimuotojen korjaus:
Tarkista partisiippimuotojen oikeellisuus ja käytä niitä oikein verbien yhteydessä.
Esimerkki: "olen opiskelut" → "olen opiskellut"
Aikamuodot ja verbien aspektit:
Varmista, että aikamuodot ovat oikein ja sopivat lauseen kontekstiin.
Esimerkki: "Pian haluisin kuitenkin..." → "Pian halusin kuitenkin..."
Monikon ja yksikön käyttö:
Huomioi, milloin sanat tulisi olla monikossa tai yksikössä lauseen merkityksen perusteella.
Esimerkki: "datakeruumenetelmiä" → "datankeruumenetelmiä"
Välimerkkien käyttö:
Tarkista, että välimerkit on sijoitettu oikein ja ne selkeyttävät lauseen rakennetta.
Esimerkki: "yrityksissä ihmiset ja huippuosaaminen ovat kilpailukyvyn..." → "yrityksissä; ihmiset ja huippuosaaminen ovat kilpailukyvyn..."

Korjausohjeet:

Lue jokainen lause huolellisesti ja tunnista mahdolliset virheet yllä mainittujen virhetyyppien perusteella.
Tee tarvittavat korjaukset niin, että lause on kieliopillisesti oikein ja säilyttää alkuperäisen merkityksensä.
Esitä korjatut lauseet selkeästi, merkitse alkuperäinen ja korjattu lause sekä perustelut näin:
Virheellinen lause:
"..."
perustelut lauseen virheellisyydelle:
"..."
Korjattu lause:
"..."
-"suggestion: tarvetta kielioppikorjauksille/ei korjattavaa".

Älä lisää uutta sisältöä tai jätä pois alkuperäistä merkityksellistä sisältöä. Korosta korjattava kohta tekstissä.
Varmista lopuksi, että koko teksti on sujuvaa ja ymmärrettävää  kieltä. 
"""



def load_users(filename="users.txt"):
    users=[]
    if os.path.exists(filename):
        with open(filename,"r",encoding="utf-8") as f:
            lines=f.read().strip().split("\n")
            block=[]
            for line in lines:
                line=line.strip()
                if line:
                    block.append(line)
                    if len(block)==3:
                        user_line=block[0]
                        pass_line=block[1]
                        key_line=block[2]
                        u=user_line.split(":",1)[1].strip()
                        p=pass_line.split(":",1)[1].strip()
                        k=key_line.split(":",1)[1].strip()
                        # Remove quotes from API key if present
                        if k.startswith('"') and k.endswith('"'):
                            k = k[1:-1]
                        logging.debug(f"Loaded user: {u}, API Key first 10 chars: {k[:10]}...")
                        users.append((u,p,k))
                        block=[]
    return users

USERS=load_users()
SESSIONS:Dict[str,tuple]={}

class LoginRequest(BaseModel):
    username:str
    password:str

class ProcessSectionsRequest(BaseModel):
    session_token:str
    selected_titles:List[str]
    text_for_corrections:str
    n_responses:int=1
    selected_model:str="fast"  # "fast" or "slow"

class LogErrorRequest(BaseModel):
    error:str

@app.get("/")
async def index():
    return {"message":"Backend is running. Access /static/taskpane.html."}

@app.get("/manifest.xml")
async def serve_manifest():
    path=os.path.join("static","manifest.xml")
    if os.path.isfile(path):
        return FileResponse(path,media_type="application/xml")
    raise HTTPException(status_code=404,detail="manifest.xml not found")

@app.get("/taskpane.html")
async def serve_taskpane():
    path=os.path.join("static","taskpane.html")
    if os.path.isfile(path):
        return FileResponse(path,media_type="text/html")
    raise HTTPException(status_code=404,detail="taskpane.html not found")

@app.post("/login")
async def login(login_data:LoginRequest):
    username=login_data.username
    password=login_data.password
    for(u,p,k) in USERS:
        if u==username and p==password:
            token=str(uuid.uuid4())
            SESSIONS[token]=(u,k)
            logging.debug(f"User {u} logged in. Session token: {token[:8]}..., API Key first 10 chars: {k[:10]}...")
            return {"session_token":token}
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,detail="Invalid credentials")

@app.post("/process_sections")
async def process_sections(data:ProcessSectionsRequest):
    session_token=data.session_token
    if session_token not in SESSIONS:
        raise HTTPException(status_code=401, detail="Not logged in")

    username,api_key=SESSIONS[session_token]
    logging.debug(f"Processing request for user {username}. API Key first 10 chars: {api_key[:10]}...")

    if not data.selected_titles or not data.text_for_corrections.strip():
        raise HTTPException(status_code=400, detail="No sections or text provided.")

    # If user selected "slow" => WordGrammarChecker uses "o3-mini" + reasoning_effort
    # If "fast" => gpt-4o
    chosen_model = data.selected_model  # "slow" or "fast"

    checker = WordGrammarChecker(
        api_key,
        system_prompt,
        n_responses=data.n_responses,
        chosen_model=chosen_model,
        use_discriminator=True  # Enable o3 discriminator for quality validation
    )

    try:
        results_per_response,_ = await checker.process_text(data.text_for_corrections)
        # results_per_response => [([corr...], suggestion), ...]
        output=[]
        for i,(corrs,sugg) in enumerate(results_per_response,start=1):
            output.append({
                "response_number": i,
                "corrections": corrs,
                "suggestion": sugg
            })
        return {"results": output}
    except Exception as e:
        #logging.exception("Error processing sections: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/log_error")
async def log_error(req:LogErrorRequest):
    msg=req.error or "No error provided"
    #logging.error("Client Error: "+msg)
    return {"status":"logged"}

# optional
@app.post("/apply_correction")
async def apply_correction():
    return {"status":"success"}

if __name__=="__main__":
    uvicorn.run(app,host="0.0.0.0",port=5000)
