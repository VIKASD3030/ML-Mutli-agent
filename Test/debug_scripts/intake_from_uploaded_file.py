"""
MANUAL SCRIPT — NOT a pytest test. Run it directly:
    python Test/debug_scripts/intake_from_uploaded_file.py

It constructs a REAL OpenAI() client and calls the API at module level, so it
costs money and needs network. It lived at Test/test_1.py, where the "test_"
prefix made pytest import it during COLLECTION — meaning simply listing the
test suite fired a live API call. Renamed and moved here, alongside the other
manual scripts, so that can no longer happen.

Confirms the full new intake path: RequirementAgent (grounded in a real
file) -> a real ProblemSpec pointing at that file -> load_data() ->
clean_and_profile() actually succeeding on it.
"""

import tempfile

import pandas as pd

from agents.requirement_agent import RequirementAgent
from pipeline.data_stage import clean_and_profile, load_data
from schemas.problem_spec import ProblemSpec

# Same tiny synthetic dataset as requirement_agent.py's own demo.
df = pd.DataFrame({
    "customer_id": range(50),
    "monthly_spend": [i * 3.7 for i in range(50)],
    "churned": [i % 3 == 0 for i in range(50)],
})
with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
    df.to_csv(f.name, index=False)
    temp_path = f.name

agent = RequirementAgent()
result = agent.run(
    "Predict whether a customer will churn, using this uploaded data.",
    file_path=temp_path,
)

print("RequirementAgent result:", type(result).__name__)
assert isinstance(result, ProblemSpec), f"Expected ProblemSpec, got: {result}"

loaded_df = load_data(result)
print("load_data() succeeded, shape:", loaded_df.shape)

cleaned_df, profile = clean_and_profile(loaded_df, result)
print("clean_and_profile() succeeded")
print("blocking_issue:", profile.blocking_issue)
print("row_count:", profile.row_count)