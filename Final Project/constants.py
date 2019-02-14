HASH_TO_INDEX_PATH = "databases/hash_to_index.db"
INDEX_TO_HASH_PATH = "databases/index_to_hash.db"

OUT_CITATIONS_PATH = "databases/out_citations"

ADJACENCY_PATH = "databases/adjacency"
RAW_DATA_PATH = "raw_data/"

ID_TO_INDEX_PATH = r'.\databases\idToIndex.db'
EDGES_VECTOR_PATH = r'.\databases\edges.db'
RANK_VEC_1_PATH = r'.\databases\rank1.db'
RANK_VEC_2_PATH = r'.\databases\rank2.db'

TEST_ID = "01276e6f0d18c35d5405b100f3028700a9363327"

LAMBDA = 0.15
EPSILON = 0.000001
MAX_ITERATIONS = 20


'''
Existing Databases
-------------------------------------------------------------------------------------
SQL DATABASES:

Structure: FILE, TABLE, PURPOSE, STATUS

'.\databases\idToIndex.db', "INDEXES_TEST", maps id to index, Created and Working
"databases/hash_to_index.db", "HASH_TO_INDEX", maps id to index, Created and Working
"databases/index_to_hash.db", "INDEX_TO_HASH", maps index to id, Created and Working

-------------------------------------------------------------------------------------
MMAP DATABASES:

Edge Matrix:

Contains 2 mmap structures,
Offset Vector and Edges Vector. Probably best not to touch directly 


Rank Vectors:

2 mmap structures that contain the current ranks (numbers)


'''