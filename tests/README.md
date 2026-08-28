# Test Scripts Directory

This directory contains test and demonstration scripts for the Loan Evaluator project.

The application package lives in `loan_evaluator/`; tests should import public
models and the evaluator from `loan_evaluator`.

## Scripts

### `test_project.py`
Comprehensive test suite validating:
- Data model imports and constraints
- Sample data loading and Pydantic validation
- Review object creation
- Prompt template availability
- Project file structure under `loan_evaluator/`
- Evaluation result compilation
- JSON serialization
- Financial calculations (DTI, LTV)

**Run:** `python3 tests/test_project.py`

### `demo.py`
Full system demonstration showing:
- Loan application loading
- Financial metric computation
- Simulated multi-agent evaluations
- Result compilation
- JSON export functionality

**Run:** `python3 tests/demo.py`

## Running Tests

From project root:
```bash
# Run test suite
python3 tests/test_project.py

# Run demonstration
python3 tests/demo.py
```

## What Gets Tested

✅ Data Models - All 6 Pydantic models  
✅ Sample Data - 3 realistic loan applications  
✅ Prompts - 4 specialized agent templates  
✅ Financial Calculations - DTI, LTV, monthly payments  
✅ JSON Export - Serialization and file output  
✅ Project Structure - Required `loan_evaluator/` package files present  

## Expected Output

Both scripts output detailed progress information with ✅/❌ indicators showing:
- Test/step name
- Status (passed/failed)
- Relevant metrics or results

All tests should pass with ✅ status.
