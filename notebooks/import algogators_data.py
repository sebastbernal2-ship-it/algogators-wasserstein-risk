import algogators_data
print("Contents of algogators_data:")
print([x for x in dir(algogators_data) if not x.startswith('_')])

# Just in case the import name differs from the pip package name
try:
    import algodata_wrapper
    print("\nContents of algodata_wrapper:")
    print([x for x in dir(algodata_wrapper) if not x.startswith('_')])
except ImportError:
    pass
