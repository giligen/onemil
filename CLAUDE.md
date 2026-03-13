# CRITICAL - ALWAYS READ FIRST
Since you have a memory of a chicken, you MUST stop every 10min and re-read CLAUDE.md AND DO THIS before every 10th prompt -- this is a must!!!

# Project: OneMil - Day Trading System

Real-time stock scanner + automated trading system targeting Ross Cameron's momentum day trading strategy.

## Goals
1. Real-time stock scanner (gap ups, high relative volume, low float, $2-$20)
2. Automated paper trading via Alpaca
3. Go live

# Code Quality
* When writing code you should behave as if you are Linus Torvalds -- partitioning, modular code, reusable code, extract common pieces to accessors, use meaningful names like Linus would
* TDD approach -- never assume that what you wrote will work. use a tdd approach to test it
* Code coverage MUST trend at ~90%, must! don't tell me you are done before code-coverage is complete
* Always validate everything you wrote via running the specific unit-test that is located in the appropriate tests directory
* Always validate everything you wrote via running a system test to ensure you didn't break anything and that the implemented functionality works
* Use Verbose/Debug flags for extra logging for the sake of debugging. Don't guess issues, find in the logging the root-cause
* Always solve the root cause, never apply work-arounds
* Instead of writing 10s of bespoke scripts, strive to use the main code models with specific flags
* Always push to github
* Always keep dependency installation file up-to-date

# Integration Testing for Multi-Component Flows

**MANDATORY**: When implementing features that involve data flowing through multiple components, you MUST write integration tests that validate the FULL end-to-end flow.

## When Integration Tests Are Required

Integration tests are REQUIRED for any flow involving:
- Database operations (save -> retrieve -> use)
- Data transformations across boundaries (JSON <-> dict, serialization/deserialization)
- Multi-step processes (input -> processing -> output)
- External API calls -> internal processing -> storage
- Configuration/state that affects multiple components

## What Integration Tests Must Cover

Integration tests MUST validate:
1. **Data integrity through the entire pipeline** - verify data format at EACH boundary
2. **Type transformations** - if data is serialized/deserialized, test both directions
3. **Edge cases** - empty data, missing fields, null values
4. **Actual component interactions** - use REAL instances, not mocks
5. **The complete flow** - from entry point to final destination

## Coverage Rule

- Unit tests: Test components in isolation (mocked dependencies)
- Integration tests: Test components working together (real dependencies)
- System tests: Test entire system end-to-end (real environment)

**You are NOT done** until you have all three levels for complex flows.

# CRITICAL: Testing & Deployment Protocol

**NEVER commit changes to production without testing first!**

## Required Testing Steps for External API Changes:
1. **Research API documentation** - Don't assume capabilities, verify them
2. **Create separate test file** - Build/test in isolation
3. **Test with real API** - Unit tests with mocks are NOT sufficient for API integration
4. **Verify success** - Check logs for actual success, not just "no errors"
5. **System test** - Run full cycle
6. **Monitor for errors** - Grep output for ERROR/exception before committing
7. **Only then commit** - If all tests pass

## Post-Incident Protocol:
1. **Immediately revert** broken code to restore production
2. **Document** what broke and why
3. **Commit revert** with clear explanation
4. **Don't rush the fix** - Take time to do it properly with testing

**Remember: Breaking production wastes more time than proper testing takes!**

# Code Standards

* Add docstrings to all functions
* Keep MD files up-to-date per model
* Keep Readme.md file up-to-date
* Unicode and emojis are supported (logging handlers must use UTF-8 encoding on Windows)
* Code should be verbose enough to show progress throughout long processes
* Use git source control and push to master every time I confirm things are working well
* Use descriptive meaningful names always
* No tests should ever be failing, always fix the core issue and don't work around it
* All Errors must be reported, e.g., missing API Keys and execution should break
* Always document latest architecture in readme.md and keep it up-to-date
* **CRITICAL: All fallback code paths MUST log ERROR or WARNING** - Silent failures hide bugs. Every fallback (try/except, if/else with defaults, .get() with fallback values) MUST explain WHY it triggered via logger.error() or logger.warning()

# System-in-dev

* Assume DB might be locked by other process that are running in parallel to you
* When building batch processors always make them verbose to show progress
* Unit tests -> Mock external APIs
* Integration tests -> Use real APIs (or testnet/paper)
* Production code -> Never includes mock logic, always real implementations

# CRITICAL: MagicMock() MUST Use spec= Parameter

**The Problem**: `MagicMock()` without `spec=` HIDES bugs by returning `MagicMock` for ANY attribute access, even non-existent ones.

## MANDATORY RULES FOR MagicMock():

### 1. ALWAYS Use spec= for Domain Classes
```python
# BAD - Hides AttributeErrors
executor = MagicMock()

# GOOD - Catches interface violations
executor = MagicMock(spec=AlpacaExecutor)
```

### 2. Use AsyncMock for Async Classes
```python
# BAD
notifier = MagicMock()

# GOOD
notifier = AsyncMock(spec=TelegramNotifier)
```

### 3. External SDK Objects Are OK Without spec=
```python
# OK - External SDK objects, not our domain
mock_order = MagicMock()  # Alpaca Order object, OK without spec
```

### 4. Use conftest.py Fixtures
Pre-configure fixtures with spec= in `tests/conftest.py`.

# Interactive Sessions
* I'm here for you to answer questions and clarify ambiguous points/logic
* **Bug Prevention Protocol**:
  - Whenever there's a bug, write BOTH unit tests AND integration tests
  - Unit test: Isolate the specific component that failed
  - Integration test: Validate the full data flow that exposed the bug
  - This ensures bugs can NEVER happen again at any level
