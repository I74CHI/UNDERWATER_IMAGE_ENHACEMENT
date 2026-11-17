# AquaEnhance Project Presentation: Questions and Answers

This guide is written for someone who is new to programming libraries. The answers are phrased so you can use them in a presentation, then explain the details if your audience asks follow-up questions.

## 1. Project introduction

### Q1. What is AquaEnhance?

**Answer:** AquaEnhance is a web application that improves the appearance of underwater photographs. A user uploads an image, the backend processes it, and the website shows the original image, the enhanced image, and several comparison results. The user can also rate the result.

**In this project:** The Python backend is primarily in `backend/main.py`. It receives requests from the website, validates and stores uploaded files, calls the image-processing service, records information in a database, and sends result URLs back to the website.

### Q2. What problem is the project trying to solve?

**Answer:** Underwater photographs can look blue or green, dark, low in contrast, or less colourful because water absorbs and scatters light. AquaEnhance applies image-processing methods to make the image clearer and more visually balanced.

**Important point:** Enhancement changes how an image looks; it cannot guarantee recovery of details that were never captured by the camera.

### Q3. What happens when someone uses the website?

**Answer:** The user chooses an image. The website sends it to the backend. The backend checks the image, saves it, processes it, stores a database record, and returns links and quality information. The website then displays the results and lets the user submit a rating.

**Simple flow:**

```text
User's browser
    -> POST /enhance with an image
    -> backend validates and saves the image
    -> image enhancement service processes it
    -> database stores image details
    -> backend returns image URLs and metrics
    -> browser displays results
    -> POST /feedback stores a rating
```

### Q4. What are the main parts of the project?

**Answer:** The main parts are:

- **Frontend:** The website the user sees, implemented with JavaScript and React/Next.js.
- **Backend/API:** The Python application that receives website requests; it uses FastAPI.
- **Image-processing service:** The code that performs enhancement and comparisons.
- **Machine-learning code:** The U-Net model and patch-processing code used when a trained model is available.
- **Database:** Stores image metadata and feedback ratings.
- **File storage:** Stores the actual uploaded and generated image files.

**Why separate these parts?** Each part has a clear job. For example, the API handles communication and validation, while the enhancement service handles image operations.

### Q5. What does “backend” mean?

**Answer:** The backend is the part of an application that runs on the server, not in the user's browser. It receives requests, applies rules, works with files and databases, and sends responses to the frontend.

**In this project:** `backend/main.py` is the entry point for the web API. It defines routes such as `/enhance`, `/feedback`, and `/health`.

## 2. Libraries and basic terms

### Q6. What is a programming library?

**Answer:** A library is reusable code written by other developers. Instead of building every feature from scratch, a project can use a library that already knows how to perform a common task.

**Example:** FastAPI provides tools for defining web addresses and accepting requests. Pillow provides tools for opening and saving image files.

### Q7. What is FastAPI, and why is it used here?

**Answer:** FastAPI is a Python library for creating web APIs. An API is a way for two programs to communicate. FastAPI lets this backend define web addresses, receive uploaded files or JSON data, validate some data, and return responses.

**In this project:** `FastAPI(...)` creates the AquaEnhance application. Decorators such as `@app.post("/enhance")` connect a Python function to a web address and HTTP method.

### Q8. What is an endpoint or route?

**Answer:** An endpoint is a particular web address that performs a particular job. A route is the rule that connects that address to the code that handles it.

**In this project:**

- `GET /` serves the website page when the frontend file is present.
- `GET /health` checks that the API can respond and reports database connectivity.
- `POST /enhance` accepts and enhances an image.
- `POST /feedback` stores a rating.
- `/storage/...` serves saved image files.

`GET` is commonly used to ask for information. `POST` is commonly used to send data to the server.

### Q9. What is Pydantic, and how is it used?

**Answer:** Pydantic is a Python library that checks and describes data. It helps ensure that incoming or outgoing data has the expected fields and types.

**In this project:** `FeedbackRequest` describes the feedback JSON. It requires an integer `image_id` and a `rating` from 1 to 5. `FeedbackResponse` describes the successful response. FastAPI uses these models to validate input and format the response.

### Q10. What is an HTTP request, and what is an HTTP response?

**Answer:** A request is a message a browser sends to a server, such as “please enhance this file.” A response is the server's reply, such as “here are the image links” or “the file is too large.”

**In this project:** The browser sends the image to `/enhance`. The backend returns JSON containing an image ID, image URLs, processing details, and comparison results. If a rule is broken, the backend returns an error response with a suitable status code and message.

### Q11. What are HTTP status codes?

**Answer:** Status codes are numbers that tell the browser whether a request worked. For example, `200` means success, `400` means the request was invalid, `404` means something was not found, and `413` means the request content was too large.

**In this project:** Invalid image formats produce a `400` error, an upload over 5 MB produces `413`, a feedback rating for an unknown image produces `404`, and image-processing or database failures produce `500`.

### Q12. What do `File`, `UploadFile`, and `python-multipart` do?

**Answer:** `UploadFile` is FastAPI's way to represent a file sent in a web request. `File(...)` tells FastAPI that a route expects a file upload. The `python-multipart` package helps FastAPI read the form format commonly used for file uploads.

**In this project:** The browser sends the image as a multipart form field named `file`. FastAPI provides it to `enhance_image` as an `UploadFile`.

### Q13. What do the Python standard-library imports do?

**Answer:** Python includes useful modules without needing to install extra packages:

- `os` reads environment settings and works with operating-system paths.
- `sys` lets this program add the project folder to Python's import search path.
- `uuid` creates a hard-to-guess unique value for file names.
- `json` converts processing information to and from JSON text.
- `pathlib.Path` represents and manipulates file and folder paths.
- `typing.Optional` describes an optional type, though it is imported but not used in this file.

### Q14. What are NumPy and Pillow used for?

**Answer:** NumPy is a library for working with arrays of numbers. A digital image can be represented as an array of pixel values. Pillow, also called PIL in imports, opens and saves common image formats.

**In this project:** The enhancement service opens images with Pillow and converts them into NumPy arrays so image-processing or machine-learning code can work with their pixel values.

### Q15. What are OpenCV and scikit-image used for?

**Answer:** OpenCV is a library with many computer-vision and image-processing operations. Scikit-image is another image-processing library, including tools for image-quality measurements.

**In this project:** The evaluation code uses OpenCV for operations such as colour-space conversion and edge detection. It uses scikit-image for reference-based measures such as SSIM and PSNR. The backend's baseline methods and quality metrics are implemented in `ml/evaluate.py`.

### Q16. What are TensorFlow and Keras?

**Answer:** TensorFlow is a machine-learning framework. Keras is its high-level interface for building, loading, and running neural-network models. A neural network is a computer model that learns patterns from examples.

**In this project:** If a trained model file exists at `ml/model/best_model.keras`, the inference code loads it and uses it to process image patches. The requirements file lists TensorFlow, but the expected model file is not present in the workspace at the time this guide was prepared. The service therefore uses its optical-restoration fallback in the current workspace.

### Q17. What are `pyodbc` and `mysql-connector-python`?

**Answer:** These are database-driver libraries. A driver lets Python communicate with a database server using that database's connection protocol.

**In this project:** `database/db.py` chooses `pyodbc` for Microsoft SQL Server, or `mysql.connector` for MySQL, based on the `DB_DRIVER` environment setting. SQL Server is the default in the code.

### Q18. What is Uvicorn?

**Answer:** Uvicorn is a program that runs an ASGI web application. ASGI is a standard way for Python web servers and web applications to communicate.

**In this project:** FastAPI defines the API, and Uvicorn can run the FastAPI application so a browser can connect to it. Uvicorn is listed in `requirements.txt`.

### Q19. What is CORS, and why is it enabled?

**Answer:** CORS means Cross-Origin Resource Sharing. Browsers use it to control whether a webpage from one address is allowed to contact a server at another address.

**In this project:** The frontend and backend may run on different local ports, so the backend adds FastAPI's CORS middleware and allows the configured frontend origins. The default origins are local addresses on port 3000.

## 3. How `backend/main.py` is organized

### Q20. What happens when `main.py` starts?

**Answer:** When Python imports this file, it reads optional settings, adds the project root to the import path, creates the storage folder if needed, configures the FastAPI application and middleware, mounts static-file folders, and registers the API routes.

**Why this matters:** Some setup happens when the application starts, not each time a user uploads a file. For example, the storage directory is created once during application setup.

### Q21. How does the program read configuration?

**Answer:** It looks for a `.env` file next to `main.py`. It reads non-empty lines that are not comments and contain `=`, then adds the values to the process environment if those variables were not already set.

**In this project:** Settings include the storage folder, API base URL, allowed frontend origins, and database-driver choice. The database helper also reads database connection settings from environment variables.

**Security note for a presentation:** A `.env` file is a configuration file and can contain private database credentials. It should not be published or shown in screenshots.

### Q22. What is an environment variable?

**Answer:** An environment variable is a setting supplied to a running program from outside its source code. It allows values such as a database password or server address to change between computers without editing the Python code.

**In this project:** `os.environ.get(...)` reads a setting and can use a default value if the setting is absent. For example, the API base URL defaults to `http://localhost:8000`.

### Q23. Why does the file add the project root to `sys.path`?

**Answer:** Python searches certain folders when importing code. Adding the project root tells Python where to find project modules that are not installed as external packages.

**In this project:** This allows `main.py` to import project files such as `database.db` and `backend.services.enhancement_service`.

### Q24. Why are paths handled with `pathlib.Path`?

**Answer:** `Path` represents a file or folder in a way that is easier to combine and inspect than manually joining text strings. It also provides operations such as checking whether a path exists and deleting a file.

**In this project:** Paths are calculated relative to the location of `main.py`, so the backend can locate the project root, frontend, and image storage folder.

### Q25. How are uploaded and generated images named?

**Answer:** The backend creates a short unique ID using `uuid`, then uses that ID in names such as `orig_<id>.jpg` and `enh_<id>.jpg`. Comparison images also use the ID.

**Why use unique names?** Two uploads should not overwrite each other's files just because they had the same original filename. The user's original filename is not used as the stored filename.

### Q26. How are saved images made available to the website?

**Answer:** FastAPI mounts the storage folder at the `/storage` web address. This means a file saved in that folder can be requested using a URL under `/storage/`.

**In this project:** The API returns URLs such as `/storage/orig_...jpg` and `/storage/enh_...jpg`. The browser uses those URLs to display the files.

### Q27. How can the backend serve the frontend?

**Answer:** At the root address `/`, the backend looks for `frontend/index.html`. If it exists, the backend sends it as a file. If it does not exist, the backend returns a small JSON message saying that the API is running. It also mounts `frontend/public` at `/public` when that folder exists.

**Project detail:** The project also has a Next.js frontend. The frontend source sends API requests to the backend; the root route's `index.html` behavior is a separate fallback/serving feature.

## 4. Image enhancement: `POST /enhance`

### Q28. What information does `/enhance` expect?

**Answer:** It expects a file upload in a form field named `file`. The route is asynchronous so it can await FastAPI's file-reading operation.

**In the website:** The frontend creates a `FormData` object, adds the selected file under the name `file`, and sends it to `/enhance`.

### Q29. How does the backend check the file type?

**Answer:** It performs two checks. First, if the browser supplied a content type, the backend accepts `image/jpeg`, `image/jpg`, or `image/png`. Second, it examines the first bytes of the file: JPEG files normally start with the JPEG marker, while PNG files start with the PNG signature.

**Why use both checks?** The content type is a label supplied with the request and can be wrong. Checking the file's opening bytes gives the backend another way to confirm that the contents resemble a supported image format.

**Project detail:** The header comment in `main.py` says JPG/JPEG, but the actual validation accepts PNG too. The error message and extension logic also support PNG.

### Q30. How does the backend enforce the 5 MB limit?

**Answer:** The backend reads the uploaded contents and checks their byte count. If the file contains more than 5 × 1024 × 1024 bytes, it returns a `413` error.

**Project detail:** The whole upload is read into memory before this size check. The limit prevents processing larger files, but a future version could reject or stream very large uploads earlier to reduce memory use.

### Q31. What happens after the file passes validation?

**Answer:** The backend chooses `.jpg` or `.png` based on the detected image signature, creates names for the original, enhanced, and baseline-comparison files, and writes the uploaded bytes to the storage folder.

**In this project:** The original is saved before enhancement begins, so the enhancement service can open it by path.

### Q32. What does `process_image` do?

**Answer:** It is the function that the API calls to perform image work. The API passes it the original file path and the path where the enhanced image should be written. The service returns information such as the number of patches, processing time, method used, image dimensions, and comparison metrics.

**Why separate this from the API?** It keeps the web-request code from containing every image-processing detail. The API coordinates the task; the service handles the image pipeline.

### Q33. What is a U-Net?

**Answer:** U-Net is a type of neural network often used for image-to-image tasks. It learns to take an image as input and produce an image as output. Its name comes from the U-shaped arrangement of its contracting and expanding sections.

**In this project:** The inference code can load a trained Keras model and run it on image patches. The model path is `ml/model/best_model.keras`. The service checks whether that file exists before selecting this route.

### Q34. Why does the project split images into patches?

**Answer:** A patch is a smaller rectangular piece of an image. Processing smaller pieces can make it possible to handle large images without asking the model to process the entire image at once.

**In this project:** The patch utility uses 256-by-256 pixel patches with overlap. It pads image edges using reflected pixels so edge areas are included. When output patches are combined, overlapping areas are blended with weights to reduce visible seams.

### Q35. What happens if the trained model is not available?

**Answer:** The service uses a traditional, physics-informed optical-restoration method called `land_like_restore`. This method compensates for colour loss, balances colour channels, stretches contrast, applies local contrast enhancement, and boosts colour vibrance.

**Current workspace behavior:** There is no model file under the expected `ml/model` folder in the workspace at the time of writing. Therefore, the fallback is the active enhancement route here.

**Important distinction:** The API and frontend describe the system as using a U-Net, but the actual method depends on whether the trained model file exists and loads successfully. A presentation should explain both paths rather than claiming that every run definitely uses the neural network.

### Q36. What are the comparison methods?

**Answer:** The service creates versions of the image using several approaches so their results can be compared:

- **Histogram equalization (`hist_eq`):** Spreads pixel brightness values to increase overall contrast.
- **CLAHE:** Increases contrast in small image regions while limiting excessive changes.
- **White balance:** Adjusts colour channels based on an assumption about the average scene colour.
- **Gamma correction:** Changes brightness using a curve, which can brighten darker areas.
- **Retinex:** Uses a blurred estimate of lighting to adjust local colour and contrast.

**In this project:** These methods run in `ml/evaluate.py`. The service saves each result and calculates quality metrics for it.

### Q37. What do the quality scores mean?

**Answer:** A metric is a numerical measurement. This project returns UIQM and UCIQE, which are designed to measure aspects of underwater-image quality without requiring a perfect reference image.

- **UIQM:** Combines measures related to colourfulness, sharpness, and contrast.
- **UCIQE:** Combines colour variation, contrast, and saturation information.

**Caution:** A metric is not the same as a human opinion. A higher score does not automatically mean every person will prefer that image.

### Q38. What are PSNR, SSIM, and MSE?

**Answer:** They are image-comparison measurements that need a reference image:

- **PSNR:** Measures pixel-level error on a logarithmic scale; higher is usually better.
- **SSIM:** Estimates similarity in image structure; values closer to 1 usually mean greater similarity.
- **MSE:** Measures average squared pixel difference; lower means smaller numerical differences.

**In this project:** `compute_all_metrics` always calculates UIQM and UCIQE. It calculates PSNR, SSIM, and MSE only if a reference image is provided. The backend's current comparison calls do not provide a reference image.

### Q39. How does the backend handle enhancement errors?

**Answer:** The image-processing call is inside a `try` block. If it raises an error, the backend removes the original, enhanced, and expected comparison files that exist, then returns an HTTP `500` error with a message describing the failure.

**Why clean up files?** If enhancement did not finish, the backend tries to avoid leaving incomplete files behind.

### Q40. How is an image result saved in the database?

**Answer:** After enhancement, the backend connects to the database and inserts the original filename, enhanced filename, and processing information into the `Image` table. It commits the change and gets the new image ID.

**Important design choice:** The image pixels are saved as files. The database stores their filenames and related details, rather than storing the full image bytes inside the database.

### Q41. Why does the SQL code use different placeholders?

**Answer:** Database drivers use different symbols for a value placeholder. This code uses `?` for SQL Server and `%s` for MySQL. The actual values are passed separately from the SQL text.

**Why pass values separately?** This is the standard safer pattern for database values and avoids building SQL by directly inserting user-supplied values into the query text.

### Q42. How does the API return the enhanced results?

**Answer:** It returns JSON containing:

- `image_id`: the database ID for this upload.
- `original_url`: a URL for the uploaded original image.
- `enhanced_url`: a URL for the main enhanced image.
- `processing_info`: details such as method, dimensions, elapsed time, and patch count.
- `comparisons`: each comparison method's name, image URL, and metrics.

The website reads this JSON and uses the URLs to display the image results.

## 5. Health check and feedback

### Q43. What does `GET /health` check?

**Answer:** It attempts to connect to the database, run the simple query `SELECT 1`, and close the connection. It returns an API status and a separate database status.

**Project detail:** Even if the database connection fails, the response's top-level `status` remains `"ok"` while the `database` field describes the problem. So the frontend or operator should inspect the database field too; `"ok"` alone does not mean the database is healthy.

### Q44. How does feedback submission work?

**Answer:** The frontend sends an `image_id` and a star rating as JSON to `POST /feedback`. Pydantic checks that the rating is from 1 to 5. The backend checks that the image ID exists, inserts the rating into the `Feedback` table, commits it, and returns a feedback ID and success message.

**If the image ID does not exist:** The backend returns `404`. If the rating is outside the permitted range or has the wrong type, FastAPI/Pydantic rejects the request before the route can store it.

### Q45. What is the relationship between the `Image` and `Feedback` tables?

**Answer:** Each image has an ID in the `Image` table. A feedback record stores that image ID, so the application can tell which image was rated. The database schema also defines a foreign-key relationship between the two tables.

**Simple explanation:** The image table is the parent record; feedback refers to it.

## 6. Reliability, design choices, and limitations

### Q46. Why does the backend use both file storage and a database?

**Answer:** Image files can be large, so this project stores the actual pictures in a storage folder. The database stores smaller pieces of information, such as filenames, processing details, IDs, timestamps, and ratings.

**Benefit:** The database can organize and find image records without putting every image's full pixel data into database rows.

### Q47. What does `commit` mean in database code?

**Answer:** A database change may be temporary until it is committed. `commit()` tells the database to make the insert permanent.

**In this project:** The backend commits after inserting an image record and after inserting a feedback record.

### Q48. What happens if the database insert for an enhanced image fails?

**Answer:** The backend tries to delete the generated image files and returns a `500` error. This avoids keeping generated files that do not have a corresponding database record.

**Project detail:** The original file is also included in the cleanup list, so it is removed in this failure case.

### Q49. What are some current limitations or possible improvements?

**Answer:** Possible improvements include:

- Store uploads in a cloud/object-storage service instead of relying on the local server folder.
- Stream or reject oversized uploads before reading the complete file into memory.
- Add user authentication and per-user access rules if the application is deployed for multiple users.
- Add operational logging and more structured error messages.
- Make the health endpoint clearly indicate an unhealthy database at the top level.
- Keep the documentation and API description consistent about accepted formats and whether U-Net is active.
- Add tests for database failures and cleanup behavior.

These are suggestions, not claims that the current application already provides them.

### Q50. How is the application tested?

**Answer:** The project includes backend API tests and frontend end-to-end tests. API tests can check behavior such as rejecting invalid uploads, accepting valid images, and validating feedback. End-to-end tests exercise the flow through the website and API.

**How to explain tests simply:** A test gives the program known input and checks whether the result matches what we expect.

## 7. Short presentation-ready summary

### Q51. Can you summarize the project in a few sentences?

**Answer:** AquaEnhance is a web application for improving underwater images. Its frontend sends an image to a Python FastAPI backend, which checks the file, saves it, and calls a separate image-processing service. The service uses a trained U-Net when the model file is available; otherwise, it applies an optical-restoration fallback, and it also creates traditional-method comparisons with underwater quality metrics. The backend saves image information and user ratings in a database and returns image links to the website.

### Q52. What is the most important thing to explain honestly during a demonstration?

**Answer:** Explain which enhancement method actually ran. The service reports the selected method in `processing_info`. If the trained model file is absent or cannot be loaded, the system uses the optical-restoration fallback instead of U-Net. Also mention that PNG is accepted, even though an introductory comment in the API file describes only JPG/JPEG.

## Quick glossary

| Term | Plain-English meaning |
|---|---|
| API | A defined way for programs to communicate |
| Backend | Server-side code that handles application work |
| Frontend | The part of an application the user sees |
| Endpoint/route | A web address connected to a specific action |
| Library | Reusable code that provides ready-made features |
| Model | A trained program that has learned patterns from examples |
| Inference | Using a trained model to produce an output |
| Patch | A smaller rectangular piece of an image |
| Fallback | An alternative method used when the preferred method is unavailable |
| Database | An organized place to store and retrieve records |
| Schema | The design of database tables and their columns |
| JSON | A text format for sending structured data between programs |
| CORS | A browser rule controlling which websites can call a server |
| Metric | A numerical measurement used to describe some property |
| Environment variable | A setting supplied to a program from outside its source code |
| Commit | Making a database change permanent |
