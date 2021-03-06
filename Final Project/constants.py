HASH_TO_INDEX_PATH = "databases/hash_to_index.db"
INDEX_TO_HASH_PATH = "databases/index_to_hash.db"

RESEARCHERS_PATH = "databases/researchers.db"
RESEARCHER_IDS_VEC = "databases/researchers_ids"

OUT_CITATIONS_PATH = "databases/out_citations"

ADJACENCY_PATH = "databases/adjacency"
RAW_DATA_PATH = "raw_data/"

ID_TO_INDEX_PATH = r'.\databases\idToIndex.db'
EDGES_VECTOR_PATH = r'.\databases\edges.db'
RANK_VEC_1_PATH = r'.\databases\rank1.db'
RANK_VEC_2_PATH = r'.\databases\rank2.db'
STEM_TXT_VEC = r'.\databases\stemText.db'
FINAL_RANK_VEC = r'.\databases\RankVector06'

TEXT_DATA_VECTOR = "text_data_vector"

# Top K articles
BEST_TEXTS = r'.\databases\clustering\best_texts'
BEST_INDEXES = r'.\databases\clustering\best_indexes'
BEST_ARTICLE_NUM = 1000
BEST_TEXT_DATA_VEC = BEST_TEXTS + str(BEST_ARTICLE_NUM)
BEST_INDEXES_VEC = BEST_INDEXES + str(BEST_ARTICLE_NUM)

TEST_ID = "01276e6f0d18c35d5405b100f3028700a9363327"
RESEARCHER_TEST_ID = 2506899

LAMBDA = 0.15
EPSILON = 0.000001
MAX_ITERATIONS = 20
RECORDS = int(10 ** 7 * 3.918)

'''
wtf we had two constant files 
HASH_TO_INDEX_PATH = "databases/hash_to_index"
ADJACENCY_PATH = "databases/adjacency"
RAW_DATA_PATH = "raw_data/"

ID_TO_INDEX_PATH = r'.\databases\idToIndex.db'
EDGES_VECTOR_PATH = r'.\databases\edges.db'
RANK_VEC_1_PATH = r'.\databases\rank1.db'
RANK_VEC_2_PATH = r'.\databases\rank2.db'
'''

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