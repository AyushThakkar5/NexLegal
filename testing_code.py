from fastapi import FastAPI, File, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    
    app = FastAPI()
    
     app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods
        allow_headers=["*"],  # Allows all headers
    )
    
    @app.post("/upload")
    async def upload_pdf(file: UploadFile = File(...)):
        try:
            contents = await file.read()
            # Process the file (e.g., save it, analyze it, etc.)
            with open(file.filename, 'wb') as f:
                f.write(contents)
        except Exception as e:
             return {"message": "There was an error uploading the file", "error": str(e)}
        finally:
             await file.close()
        return {"filename": file.filename, "message": "File uploaded successfully"}