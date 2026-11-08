import sys
try:
    import ctcdecode
    print("ctcdecode OK")
except Exception as e:
    print("ctcdecode MISSING:", type(e).__name__, e)
