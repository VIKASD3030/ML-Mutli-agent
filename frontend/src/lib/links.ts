/** Where the Streamlit test-harness UI (streamlit_app.py) is served. */
export const STREAMLIT_URL: string =
  import.meta.env.VITE_STREAMLIT_URL ?? 'http://localhost:8501'
