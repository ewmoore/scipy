#ifndef __COO_H__
#define __COO_H__

#include <algorithm>

/*
 * Compute B = A for COO matrix A, CSR matrix B
 *
 *
 * Input Arguments:
 *   I  n_row      - number of rows in A
 *   I  n_col      - number of columns in A
 *   I  nnz        - number of nonzeros in A
 *   I  Ai[nnz(A)] - row indices
 *   I  Aj[nnz(A)] - column indices
 *   T  Ax[nnz(A)] - nonzeros
 * Output Arguments:
 *   I Bp  - row pointer
 *   I Bj  - column indices
 *   T Bx  - nonzeros
 *
 * Note:
 *   Output arrays Bp, Bj, and Bx must be preallocated
 *
 * Note: 
 *   Input:  row and column indices *are not* assumed to be ordered
 *           
 *   Note: duplicate entries are carried over to the CSR represention
 *
 *   Complexity: Linear.  Specifically O(nnz(A) + max(n_row,n_col))
 * 
 */
template <class I, class T>
void coo_tocsr(const I n_row,
               const I n_col,
               const I nnz,
               const I Ai[],
               const I Aj[],
               const T Ax[],
                     I Bp[],
                     I Bj[],
                     T Bx[])
{
    //compute number of non-zero entries per row of A 
    std::fill(Bp, Bp + n_row, 0);

    for (I n = 0; n < nnz; n++){            
        Bp[Ai[n]]++;
    }

    //cumsum the nnz per row to get Bp[]
    for(I i = 0, cumsum = 0; i < n_row; i++){     
        I temp = Bp[i];
        Bp[i] = cumsum;
        cumsum += temp;
    }
    Bp[n_row] = nnz; 

    //write Aj,Ax into Bj,Bx
    for(I n = 0; n < nnz; n++){
        I row  = Ai[n];
        I dest = Bp[row];

        Bj[dest] = Aj[n];
        Bx[dest] = Ax[n];

        Bp[row]++;
    }

    for(I i = 0, last = 0; i <= n_row; i++){
        I temp = Bp[i];
        Bp[i]  = last;
        last   = temp;
    }

    //now Bp,Bj,Bx form a CSR representation (with possible duplicates)
}

/*
 * Compute B += A for COO matrix A, dense matrix B
 *
 * Input Arguments:
 *   I  n_row           - number of rows in A
 *   I  n_col           - number of columns in A
 *   I  nnz             - number of nonzeros in A
 *   I  Ai[nnz(A)]      - row indices
 *   I  Aj[nnz(A)]      - column indices
 *   T  Ax[nnz(A)]      - nonzeros 
 *   T  Bx[n_row*n_col] - dense matrix
 *
 */
template <class I, class T>
void coo_todense(const I n_row,
                 const I n_col,
                 const I nnz,
                 const I Ai[],
                 const I Aj[],
                 const T Ax[],
                       T Bx[],
		 int fortran)
{
    if (!fortran) {
        for(I n = 0; n < nnz; n++){
            Bx[ (npy_intp)n_col * Ai[n] + Aj[n] ] += Ax[n];
        }
    }
    else {
        for(I n = 0; n < nnz; n++){
            Bx[ (npy_intp)n_row * Aj[n] + Ai[n] ] += Ax[n];
        }
    }
}


/*
 * Compute Y = alpha*A*X + beta*Y for COO matrix A and dense vectors X,Y
 * and scalars alpha and beta
 *
 *
 * Input Arguments:
 *   I  nnz           - number of nonzeros in A
 *   I  Ai[nnz]       - row indices
 *   I  Aj[nnz]       - column indices
 *   T  Ax[nnz]       - nonzero values
 *   T  alpha         - scalar factor multiplying A*X
 *   T  Xx[n_col]     - input vector
 *   T  beta          - scalar factor multiplying Y
 *
 * Output Arguments:
 *   T  Yx[n_row]     - output vector
 *
 * Notes:
 *   Output array Yx must be preallocated
 *
 *   Complexity: Linear.  Specifically O(nnz(A))
 *
 */
template <class I, class T>
void coo_matvec(const I n_row,
                const I nnz,
                const I Ai[],
                const I Aj[],
                const T Ax[],
                const T alpha,
                const T Xx[],
                const T beta,
                      T Yx[])
{
    scal(n_row, beta, Yx);
    for(I n = 0; n < nnz; n++){
        Yx[Ai[n]] += alpha * Ax[n] * Xx[Aj[n]];
    }
}

#endif
