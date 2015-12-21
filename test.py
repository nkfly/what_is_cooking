from scipy.sparse import csr_matrix, coo_matrix

def convert(term_dict):
    ''' Convert a dictionary with elements of form ('d1', 't1'): 12 to a CSR type matrix.
    The element ('d1', 't1'): 12 becomes entry (0, 0) = 12.
    * Conversion from 1-indexed to 0-indexed.
    * d is row
    * t is column.
    '''
    # Create the appropriate format for the COO format.
    data = []
    row = []
    col = []
    for k, v in term_dict.items():
        r = int(k[0][1:])
        print 'r', k[0], k[0][1:]
	c = int(k[1][1:])
        data.append(v)
        row.append(r-1)
        col.append(c-1)
    # Create the COO-matrix
    coo = coo_matrix((data,(row,col)))
    # Let Scipy convert COO to CSR format and return
    return csr_matrix(coo)

if __name__=='__main__':
    doc_term_dict = { ('d1','t1'): 12,             \
                ('d2','t3'): 10,             \
                ('d3','t2'):  5              \
                }   
    print(convert(doc_term_dict))
