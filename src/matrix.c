#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid. Note that the matrix is in row-major order.
 */
double get(matrix *mat, int row, int col) {
    // Task 1.1 TODO
    int i = mat->cols * row + col;
    return (mat->data)[i];
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid. Note that the matrix is in row-major order.
 */
void set(matrix *mat, int row, int col, double val) {
    // Task 1.1 TODO
    int i = mat->cols * row + col;
    mat->data[i] = val;
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    // Task 1.2 TODO
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
    if (rows <= 0 || cols <= 0) {
        return -1;
    }
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    matrix* m = (matrix*) malloc(sizeof(matrix));
    if (m == NULL) {
        return -2;
    }

    // 3. Allocate space for the matrix data, initializing all entries to be 0. Return -2 if allocating memory failed.
    int size = rows * cols;
    double* data = (double*) calloc(size, sizeof(double));
    if (data == NULL) {
        return -2;
    }
    m->data = data;

    // 4. Set the number of rows and columns in the matrix struct according to the arguments provided.
    m->cols = cols;
    m->rows = rows;

    // 5. Set the `parent` field to NULL, since this matrix was not created from a slice.
    m->parent = NULL;

    // 6. Set the `ref_cnt` field to 1.
    m->ref_cnt = 1;

    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
    *mat = m;
    
    // 8. Return 0 upon success.
    return 0;
}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice and has no existing slices,
 * or that you free `mat->parent->data` if `mat` is the last existing slice of its parent matrix and its parent
 * matrix has no other references (including itself).
 */
void deallocate_matrix(matrix *mat) {
    // Task 1.3 TODO
    // HINTS: Follow these steps.
    // 1. If the matrix pointer `mat` is NULL, return.
    if (mat == NULL) {
        return;
    }
    // 2. If `mat` has no parent: decrement its `ref_cnt` field by 1. If the `ref_cnt` field becomes 0, then free `mat` and its `data` field.
    if (mat->parent == NULL) {
        mat->ref_cnt--;
        if (mat->ref_cnt == 0) {
            free(mat->data);
            free(mat);
        }
    } 
    // 3. Otherwise, recursively call `deallocate_matrix` on `mat`'s parent, then free `mat`.
    else {
        deallocate_matrix(mat->parent);
        free(mat);
    }
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`
 * and the reference counter for `from` should be incremented. Lastly, do not forget to set the
 * matrix's row and column values as well.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 * NOTE: Here we're allocating a matrix struct that refers to already allocated data, so
 * there is no need to allocate space for matrix data.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    // Task 1.4 TODO
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
    if (rows <= 0 || cols <= 0) {
        return -1;
    }

    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    matrix* m = (matrix*) malloc(sizeof(matrix));
    if (m == NULL) {
        return -2;
    }

    // 3. Set the `data` field of the new struct to be the `data` field of the `from` struct plus `offset`.
    m->data = (from->data)+offset;

    // 4. Set the number of rows and columns in the new struct according to the arguments provided.
    m->rows = rows;
    m->cols = cols;
    // 5. Set the `parent` field of the new struct to the `from` struct pointer.
    m->parent = from;
    // 6. Increment the `ref_cnt` field of the `from` struct by 1.
    from->ref_cnt++;
    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
    *mat = m;
    // 8. Return 0 upon success.
    return 0;
}

/*
 * Sets all entries in mat to val. Note that the matrix is in row-major order.
 */
void fill_matrix(matrix *mat, double val) {
    int size = mat->rows * mat->cols;
    int end = size/4*4;
    int i;
    #pragma omp parallel for
    for (i = 0; i < end; i+=4) {
        __m256d values =  _mm256_set1_pd(val);
        _mm256_storeu_pd((mat->data + i), values);
    }
    for (; i < size; i++) {
        (mat->data)[i] = val;
    }
}
void fill_matrix_naive(matrix *mat, double val) {
    // Task 1.5 TODO
    int size = mat->rows * mat->cols;
    for (int i = 0; i < size; i++) {
        set(mat, i/mat->cols, i % mat->cols, val);
    }
}


/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int abs_matrix(matrix *result, matrix *mat) {
    int size = mat->rows * mat->cols;
    int unroll;
    __m256d mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF)); 
    if (size >= 10000) {
        unroll = 16;
        int end = (size >> 4) << 4;

        #pragma omp parallel for 
        for (int i = 0; i < end; i+=16) {
            // __m256d values1 = _mm256_loadu_pd(mat->data + i);
            // __m256d values2 = _mm256_loadu_pd(mat->data + i+4);
            // __m256d values3 = _mm256_loadu_pd(mat->data + i+8);
            // __m256d values4 = _mm256_loadu_pd(mat->data + i+12);

            // _mm256_storeu_pd(result->data + i, _mm256_and_pd(values1, mask));
            // _mm256_storeu_pd(result->data + i+4, _mm256_and_pd(values2, mask));
            // _mm256_storeu_pd(result->data + i+8, _mm256_and_pd(values3, mask));
            // _mm256_storeu_pd(result->data + i+12, _mm256_and_pd(values4, mask));

            _mm256_storeu_pd(result->data + i, _mm256_and_pd(_mm256_loadu_pd(mat->data + i), mask));
            _mm256_storeu_pd(result->data + i+4, _mm256_and_pd(_mm256_loadu_pd(mat->data + i+4), mask));
            _mm256_storeu_pd(result->data + i+8, _mm256_and_pd(_mm256_loadu_pd(mat->data + i+8), mask));
            _mm256_storeu_pd(result->data + i+12, _mm256_and_pd(_mm256_loadu_pd(mat->data + i+12), mask));            
            // _mm256_storeu_pd(result->data + i+16, _mm256_and_pd(_mm256_loadu_pd(mat->data + i+16), mask));            
            // _mm256_storeu_pd(result->data + i+20, _mm256_and_pd(_mm256_loadu_pd(mat->data + i+20), mask));            
            // _mm256_storeu_pd(result->data + i+24, _mm256_and_pd(_mm256_loadu_pd(mat->data + i+24), mask));            
            // _mm256_storeu_pd(result->data + i+28, _mm256_and_pd(_mm256_loadu_pd(mat->data + i+28), mask));            
        }
        
        for (int i = end; i < size; i++) {
            (result->data)[i] = fabs((mat->data)[i]);
        }
    } else {
        int end = (size >> 3) << 3;
        int i;
        for (i = 0; i < end; i+=8) {
            //(result->data)[i] = fabs((mat->data)[i]);
            // __m256d values1 = ;
            _mm256_storeu_pd(result->data + i, _mm256_and_pd(_mm256_loadu_pd(mat->data + i), mask));
            _mm256_storeu_pd(result->data + i+4, _mm256_and_pd(_mm256_loadu_pd(mat->data + i+4), mask));
        }
        for (; i < size; i++) {
            (result->data)[i] = fabs((mat->data)[i]);
        }
    }
    return 0;
}
int abs_matrix_par(matrix *result, matrix *mat) {
    int size = mat->rows * mat->cols;
    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        (result->data)[i] = fabs((mat->data)[i]);
    }
    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int neg_matrix(matrix *result, matrix *mat) {
    // Task 1.5 TODO
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.5 TODO
    int size = result->rows * result->cols;
    int unroll = 16;
    if (size < 100) {
        int i;
        for (i = 0; i < size; i++) {
            result->data[i] = mat1->data[i] + mat2->data[i];
        } 
    } else {
        int end = size/unroll*unroll;
        __m256d m1_1, m1_2, m1_3, m1_4, m2_1, m2_2, m2_3, m2_4;
        #pragma omp parallel for private(m1_1, m1_2, m1_3, m1_4, m2_1, m2_2, m2_3, m2_4)
        for (int i = 0; i < end; i+=unroll) {
            m1_1 = _mm256_loadu_pd(mat1->data+i);
            m1_2 = _mm256_loadu_pd(mat1->data+i+4);
            m1_3 = _mm256_loadu_pd(mat1->data+i+8);
            m1_4 = _mm256_loadu_pd(mat1->data+i+12);
            
            m2_1 = _mm256_loadu_pd(mat2->data+i);
            m2_2 = _mm256_loadu_pd(mat2->data+i+4);
            m2_3 = _mm256_loadu_pd(mat2->data+i+8);
            m2_4 = _mm256_loadu_pd(mat2->data+i+12);
            
            m1_1 = _mm256_add_pd (m1_1, m2_1);
            m1_2 = _mm256_add_pd (m1_2, m2_2);
            m1_3 = _mm256_add_pd (m1_3, m2_3);
            m1_4 = _mm256_add_pd (m1_4, m2_4);

            _mm256_storeu_pd((result->data + i), m1_1);
            _mm256_storeu_pd((result->data + i+4), m1_2);
            _mm256_storeu_pd((result->data + i+8), m1_3);
            _mm256_storeu_pd((result->data + i+12), m1_4);
        }
        for (int i = end; i < size; i+=1) {
            result->data[i] = mat1->data[i] + mat2->data[i];
        }
    }

    return 0;
}
int add_matrix_naive(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.5 TODO
    int size = result->rows * result->cols;
    for (int i = 0; i < size; i++) {
        result->data[i] = mat1->data[i] + mat2->data[i];
    }
    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.5 TODO
}

void transpose_blocking(int r, int c, int blocksize, double *dst, double *src) {
    int row_lim = (r+blocksize-1)/blocksize;
    int col_lim = (c+blocksize-1)/blocksize;
    for (int x = 0; x < row_lim; x++) {
        for (int y = 0; y < col_lim; y++) {
            for (int i = 0; i < blocksize; i++) {
                for (int j = 0; j < blocksize; j++) {
                    if (i + x * blocksize < r && j+ y * blocksize < c) {
                        dst[y * blocksize * r + blocksize * x + i + j*r] = 
                        src[x * blocksize * c + blocksize * y + j + i*c];
                    }                        
                }
            }
        }
    }
}

void transpose(int r, int c, double *dst, double *src) {
    if (r*c < 100) {
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                dst[i + j*r] = src[j + i*c];                      
            }
        }
    } else {
        #pragma omp parallel for
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                dst[i + j*r] = src[j + i*c];                      
            }
        } 
    }
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 * You may assume `mat1`'s number of columns is equal to `mat2`'s number of rows.
 * Note that the matrix is in row-major order.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.6 TODO
    int blocksize = 32;

    int size = result->rows*result->cols;

    int r = mat2->rows;
    int c = mat2->cols;    
    double *mat2_t = malloc(r * c * sizeof(double));
    transpose_blocking(r, c, blocksize, mat2_t, mat2->data);
    // transpose(r, c, mat2_t, mat2->data);
    // if (r*c < 100) {
    //     for (int i = 0; i < r; i++) {
    //         for (int j = 0; j < c; j++) {
    //             mat2_t[i + j*r] = mat2->data[j + i*c];                      
    //         }
    //     }
    // } else {
    //     #pragma omp parallel for
    //     for (int i = 0; i < r; i++) {
    //         for (int j = 0; j < c; j++) {
    //             mat2_t[i + j*r] = mat2->data[j + i*c];                      
    //         }
    //     } 
    // }

    r = result->rows;
    c = result->cols;
    int inner_dim = mat1->cols;
    double *temp = malloc(size * sizeof(double));
    int unroll;
    if (size < 400) {
        double sum;
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                sum = 0;
                for (int k = 0; k < inner_dim; k++) {
                    sum += mat1->data[inner_dim * i + k] * mat2_t[inner_dim * j + k];
                }
                temp[i*c + j] = sum;
            }
        }
        for (int i = 0; i < size; i++) {
            result->data[i] = temp[i];
        }
    } else {

        if (size <= 1000) {
            __m256d sum;
            double sums[4] = {0, 0, 0, 0};
            double sum_tail;
            // int muli, mulj;
            #pragma omp parallel for private(sums, sum_tail)
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    sum = _mm256_set1_pd(0);
                    sum_tail = 0;
                    int k;
                    for (k = 0; k < inner_dim/4*4; k+=4) {
                        __m256d m1 = _mm256_loadu_pd(mat1->data + inner_dim * i + k);
                        __m256d m2 = _mm256_loadu_pd(mat2_t + inner_dim * j + k);
                        sum = _mm256_add_pd (sum, _mm256_mul_pd (m1, m2)); 
                    }
                    for (; k < inner_dim; k++) {
                        sum_tail += mat1->data[inner_dim * i + k] * mat2_t[inner_dim * j + k];
                    }
                    _mm256_storeu_pd (sums, sum);
                    temp[i*c + j] = sum_tail + sums[0] + sums[1] + sums[2] + sums[3];
                }
            }
        } else if (size <= 10001) {
            unroll = 16;
            double sums[4];
            double sum_tail;
            int mulj; int muli;
            __m256d sum1, sum2, sum3, sum4;
            __m256d m1_1, m1_2, m1_3, m1_4;
            __m256d m2_1, m2_2, m2_3, m2_4;
            #pragma omp parallel for private(sums, sum_tail,sum1, sum2, sum3, sum4, m1_1, m1_2, m2_1, m2_2, m1_3, m1_4, m2_3, m2_4) schedule(dynamic)
            for (int i = 0; i < r; i++) {
                muli = inner_dim * i;
                for (int j = 0; j < c; j++) {
                    sum1 = _mm256_set1_pd(0);
                    sum2 = _mm256_set1_pd(0);
                    sum3 = _mm256_set1_pd(0);
                    sum4 = _mm256_set1_pd(0);
                    sum_tail = 0;
                    mulj = inner_dim * j;
                    int k = 0;
                    for (; k < inner_dim/unroll*unroll; k+=unroll) {
                        int mulik = muli+k;
                        int muljk = mulj +k;
                        m1_1 = _mm256_loadu_pd(mat1->data + mulik);
                        m1_2 = _mm256_loadu_pd(mat1->data + mulik +4);
                        m1_3 = _mm256_loadu_pd(mat1->data + mulik +8);
                        m1_4 = _mm256_loadu_pd(mat1->data + mulik +12);

                        m2_1 = _mm256_loadu_pd(mat2_t + muljk );
                        m2_2 = _mm256_loadu_pd(mat2_t + muljk +4);
                        m2_3 = _mm256_loadu_pd(mat2_t + muljk +8);
                        m2_4 = _mm256_loadu_pd(mat2_t + muljk +12);

                        sum1 = _mm256_add_pd (sum1, _mm256_mul_pd (m1_1, m2_1)); 
                        sum2 = _mm256_add_pd (sum2, _mm256_mul_pd (m1_2, m2_2)); 
                        sum3 = _mm256_add_pd (sum3, _mm256_mul_pd (m1_3, m2_3)); 
                        sum4 = _mm256_add_pd (sum4, _mm256_mul_pd (m1_4, m2_4)); 
                    }
                    // unroll = 4;
                    // for (; k < inner_dim/4*4; k+=4) {
                    //     m1_1 = _mm256_loadu_pd(mat1->data + muli + k);
                    //     m1_2 = _mm256_loadu_pd(mat2_t + mulj + k);
                    //     sum1 = _mm256_add_pd (sum1, _mm256_mul_pd (m1_1, m1_2)); 
                    // }                    
                    for (; k < inner_dim; k++) {
                        sum_tail += mat1->data[inner_dim*i + k] * mat2_t[inner_dim*j + k];
                    }
                    // sum1 = _mm256_add_pd(sum1, sum2);
                    // sum3 = _mm256_add_pd(sum3, sum4);
                    // sum1 = _mm256_add_pd(sum1, sum3);
                    _mm256_storeu_pd (sums, sum1);
                    _mm256_storeu_pd (sums+4, sum2);
                    _mm256_storeu_pd (sums+8, sum3);
                    _mm256_storeu_pd (sums+12, sum4);
                    // temp[i*c + j] = sum_tail + sums[0] + sums[1] + sums[2] + sums[3];
                    temp[i*c + j] = sum_tail + sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5] + sums[6] + sums[7];
                    // temp[i*c + j] = sums[8] + sums[9] + sums[10] + sums[11];

                    temp[i*c + j] += sums[8] + sums[9] + sums[10] + sums[11] + sums[12] + sums[13] + sums[14] + sums[15];
                }
            }
        } else {
            unroll = 16;
            double sums[16];
            double sum_tail;
            __m256d sum1, sum2, sum3, sum4;
            __m256d m1_1, m1_2, m1_3, m1_4;
            __m256d m2_1, m2_2, m2_3, m2_4;
            int tempi;
            #pragma omp parallel for private(sums, sum_tail,sum1, sum2, sum3, sum4, m1_1, m1_2, m1_3, m1_4, m2_1, m2_2, m2_3, m2_4) schedule(dynamic)
            for (int i = 0; i < r/8*8; i+=8) {
                for (int j = 0; j < c; j++) {
                    // sum1 = _mm256_set1_pd(0);
                    // sum2 = _mm256_set1_pd(0);
                    // sum3 = _mm256_set1_pd(0);
                    // sum4 = _mm256_set1_pd(0);
                    // sum_tail = 0;
                    // int k = 0;
                    // for (; k < inner_dim/unroll*unroll; k+=unroll) {
                    //     m1_1 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k);
                    //     m1_2 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k +4);
                    //     m1_3 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k +8);
                    //     m1_4 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k +12);

                    //     m2_1 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k );
                    //     m2_2 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k +4);
                    //     m2_3 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k +8);
                    //     m2_4 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k +12);

                    //     sum1 = _mm256_add_pd (sum1, _mm256_mul_pd (m1_1, m2_1)); 
                    //     sum2 = _mm256_add_pd (sum2, _mm256_mul_pd (m1_2, m2_2)); 
                    //     sum3 = _mm256_add_pd (sum3, _mm256_mul_pd (m1_3, m2_3)); 
                    //     sum4 = _mm256_add_pd (sum4, _mm256_mul_pd (m1_4, m2_4)); 
                    // }

                    // for (; k < inner_dim; k++) {
                    //     sum_tail += mat1->data[inner_dim*i + k] * mat2_t[inner_dim*j + k];
                    // }
                    // _mm256_storeu_pd (sums, sum1);
                    // _mm256_storeu_pd (sums+4, sum2);
                    // _mm256_storeu_pd (sums+8, sum3);
                    // _mm256_storeu_pd (sums+12, sum4);
                    // temp[i*c + j] = sum_tail + sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5] + sums[6] + sums[7];
                    // temp[i*c + j] += sums[8] + sums[9] + sums[10] + sums[11] + sums[12] + sums[13] + sums[14] + sums[15];
                
                    // i++;

                    // sum1 = _mm256_set1_pd(0);
                    // sum2 = _mm256_set1_pd(0);
                    // sum3 = _mm256_set1_pd(0);
                    // sum4 = _mm256_set1_pd(0);
                    // sum_tail = 0;
                    // k = 0;
                    // for (; k < inner_dim/unroll*unroll; k+=unroll) {
                    //     m1_1 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k);
                    //     m1_2 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k +4);
                    //     m1_3 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k +8);
                    //     m1_4 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k +12);

                    //     m2_1 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k );
                    //     m2_2 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k +4);
                    //     m2_3 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k +8);
                    //     m2_4 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k +12);

                    //     sum1 = _mm256_add_pd (sum1, _mm256_mul_pd (m1_1, m2_1)); 
                    //     sum2 = _mm256_add_pd (sum2, _mm256_mul_pd (m1_2, m2_2)); 
                    //     sum3 = _mm256_add_pd (sum3, _mm256_mul_pd (m1_3, m2_3)); 
                    //     sum4 = _mm256_add_pd (sum4, _mm256_mul_pd (m1_4, m2_4)); 
                    // }                  
                    // for (; k < inner_dim; k++) {
                    //     sum_tail += mat1->data[inner_dim*i + k] * mat2_t[inner_dim*j + k];
                    // }

                    // _mm256_storeu_pd (sums, sum1);
                    // _mm256_storeu_pd (sums+4, sum2);
                    // _mm256_storeu_pd (sums+8, sum3);
                    // _mm256_storeu_pd (sums+12, sum4);
                    // temp[i*c + j] = sum_tail + sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5] + sums[6] + sums[7];

                    // temp[i*c + j] += sums[8] + sums[9] + sums[10] + sums[11] + sums[12] + sums[13] + sums[14] + sums[15];
                
                    // i++;

                    sum1 = _mm256_set1_pd(0);
                    sum2 = _mm256_set1_pd(0);
                    sum3 = _mm256_set1_pd(0);
                    sum4 = _mm256_set1_pd(0);
                    sum_tail = 0;
                    int k = 0;
                    for (; k < inner_dim/unroll*unroll; k+=unroll) {
                        m1_1 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k);
                        m1_2 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k +4);
                        m1_3 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k +8);
                        m1_4 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k +12);

                        m2_1 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k );
                        m2_2 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k +4);
                        m2_3 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k +8);
                        m2_4 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k +12);

                        sum1 = _mm256_add_pd (sum1, _mm256_mul_pd (m1_1, m2_1)); 
                        sum2 = _mm256_add_pd (sum2, _mm256_mul_pd (m1_2, m2_2)); 
                        sum3 = _mm256_add_pd (sum3, _mm256_mul_pd (m1_3, m2_3)); 
                        sum4 = _mm256_add_pd (sum4, _mm256_mul_pd (m1_4, m2_4)); 
                    }

                    for (; k < inner_dim; k++) {
                        sum_tail += mat1->data[inner_dim*i + k] * mat2_t[inner_dim*j + k];
                    }

                    _mm256_storeu_pd (sums, sum1);
                    _mm256_storeu_pd (sums+4, sum2);
                    _mm256_storeu_pd (sums+8, sum3);
                    _mm256_storeu_pd (sums+12, sum4);
                    temp[i*c + j] = sum_tail + sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5] + sums[6] + sums[7];
                    temp[i*c + j] += sums[8] + sums[9] + sums[10] + sums[11] + sums[12] + sums[13] + sums[14] + sums[15];
                
                    i++;
                    sum1 = _mm256_set1_pd(0);
                    sum2 = _mm256_set1_pd(0);
                    sum3 = _mm256_set1_pd(0);
                    sum4 = _mm256_set1_pd(0);
                    sum_tail = 0;
                    k = 0;
                    for (; k < inner_dim/unroll*unroll; k+=unroll) {
                        m1_1 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k);
                        m1_2 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k +4);
                        m1_3 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k +8);
                        m1_4 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k +12);

                        m2_1 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k );
                        m2_2 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k +4);
                        m2_3 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k +8);
                        m2_4 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k +12);

                        sum1 = _mm256_add_pd (sum1, _mm256_mul_pd (m1_1, m2_1)); 
                        sum2 = _mm256_add_pd (sum2, _mm256_mul_pd (m1_2, m2_2)); 
                        sum3 = _mm256_add_pd (sum3, _mm256_mul_pd (m1_3, m2_3)); 
                        sum4 = _mm256_add_pd (sum4, _mm256_mul_pd (m1_4, m2_4)); 
                    }
                 
                    for (; k < inner_dim; k++) {
                        sum_tail += mat1->data[inner_dim*i + k] * mat2_t[inner_dim*j + k];
                    }
                    _mm256_storeu_pd (sums, sum1);
                    _mm256_storeu_pd (sums+4, sum2);
                    _mm256_storeu_pd (sums+8, sum3);
                    _mm256_storeu_pd (sums+12, sum4);
                    temp[i*c + j] = sum_tail + sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5] + sums[6] + sums[7];
                    temp[i*c + j] += sums[8] + sums[9] + sums[10] + sums[11] + sums[12] + sums[13] + sums[14] + sums[15];

                    i++;

                    sum1 = _mm256_set1_pd(0);
                    sum2 = _mm256_set1_pd(0);
                    sum3 = _mm256_set1_pd(0);
                    sum4 = _mm256_set1_pd(0);
                    sum_tail = 0;
                    k = 0;
                    for (; k < inner_dim/unroll*unroll; k+=unroll) {
                        m1_1 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k);
                        m1_2 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k +4);
                        m1_3 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k +8);
                        m1_4 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k +12);

                        m2_1 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k );
                        m2_2 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k +4);
                        m2_3 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k +8);
                        m2_4 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k +12);

                        sum1 = _mm256_add_pd (sum1, _mm256_mul_pd (m1_1, m2_1)); 
                        sum2 = _mm256_add_pd (sum2, _mm256_mul_pd (m1_2, m2_2)); 
                        sum3 = _mm256_add_pd (sum3, _mm256_mul_pd (m1_3, m2_3)); 
                        sum4 = _mm256_add_pd (sum4, _mm256_mul_pd (m1_4, m2_4)); 
                    }

                    for (; k < inner_dim; k++) {
                        sum_tail += mat1->data[inner_dim*i + k] * mat2_t[inner_dim*j + k];
                    }
                    _mm256_storeu_pd (sums, sum1);
                    _mm256_storeu_pd (sums+4, sum2);
                    _mm256_storeu_pd (sums+8, sum3);
                    _mm256_storeu_pd (sums+12, sum4);
                    temp[i*c + j] = sum_tail + sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5] + sums[6] + sums[7];
                    temp[i*c + j] += sums[8] + sums[9] + sums[10] + sums[11] + sums[12] + sums[13] + sums[14] + sums[15];
                
                    i++;

                    sum1 = _mm256_set1_pd(0);
                    sum2 = _mm256_set1_pd(0);
                    sum3 = _mm256_set1_pd(0);
                    sum4 = _mm256_set1_pd(0);
                    sum_tail = 0;
                    k = 0;
                    for (; k < inner_dim/unroll*unroll; k+=unroll) {
                        m1_1 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k);
                        m1_2 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k +4);
                        m1_3 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k +8);
                        m1_4 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k +12);

                        m2_1 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k );
                        m2_2 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k +4);
                        m2_3 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k +8);
                        m2_4 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k +12);

                        sum1 = _mm256_add_pd (sum1, _mm256_mul_pd (m1_1, m2_1)); 
                        sum2 = _mm256_add_pd (sum2, _mm256_mul_pd (m1_2, m2_2)); 
                        sum3 = _mm256_add_pd (sum3, _mm256_mul_pd (m1_3, m2_3)); 
                        sum4 = _mm256_add_pd (sum4, _mm256_mul_pd (m1_4, m2_4)); 
                    }                  
                    for (; k < inner_dim; k++) {
                        sum_tail += mat1->data[inner_dim*i + k] * mat2_t[inner_dim*j + k];
                    }

                    _mm256_storeu_pd (sums, sum1);
                    _mm256_storeu_pd (sums+4, sum2);
                    _mm256_storeu_pd (sums+8, sum3);
                    _mm256_storeu_pd (sums+12, sum4);
                    temp[i*c + j] = sum_tail + sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5] + sums[6] + sums[7];

                    temp[i*c + j] += sums[8] + sums[9] + sums[10] + sums[11] + sums[12] + sums[13] + sums[14] + sums[15];
                
                    i++;

                    sum1 = _mm256_set1_pd(0);
                    sum2 = _mm256_set1_pd(0);
                    sum3 = _mm256_set1_pd(0);
                    sum4 = _mm256_set1_pd(0);
                    sum_tail = 0;
                    k = 0;
                    for (; k < inner_dim/unroll*unroll; k+=unroll) {
                        m1_1 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k);
                        m1_2 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k +4);
                        m1_3 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k +8);
                        m1_4 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k +12);

                        m2_1 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k );
                        m2_2 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k +4);
                        m2_3 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k +8);
                        m2_4 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k +12);

                        sum1 = _mm256_add_pd (sum1, _mm256_mul_pd (m1_1, m2_1)); 
                        sum2 = _mm256_add_pd (sum2, _mm256_mul_pd (m1_2, m2_2)); 
                        sum3 = _mm256_add_pd (sum3, _mm256_mul_pd (m1_3, m2_3)); 
                        sum4 = _mm256_add_pd (sum4, _mm256_mul_pd (m1_4, m2_4)); 
                    }

                    for (; k < inner_dim; k++) {
                        sum_tail += mat1->data[inner_dim*i + k] * mat2_t[inner_dim*j + k];
                    }

                    _mm256_storeu_pd (sums, sum1);
                    _mm256_storeu_pd (sums+4, sum2);
                    _mm256_storeu_pd (sums+8, sum3);
                    _mm256_storeu_pd (sums+12, sum4);
                    temp[i*c + j] = sum_tail + sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5] + sums[6] + sums[7];
                    temp[i*c + j] += sums[8] + sums[9] + sums[10] + sums[11] + sums[12] + sums[13] + sums[14] + sums[15];
                
                    i++;
                    sum1 = _mm256_set1_pd(0);
                    sum2 = _mm256_set1_pd(0);
                    sum3 = _mm256_set1_pd(0);
                    sum4 = _mm256_set1_pd(0);
                    sum_tail = 0;
                    k = 0;
                    for (; k < inner_dim/unroll*unroll; k+=unroll) {
                        m1_1 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k);
                        m1_2 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k +4);
                        m1_3 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k +8);
                        m1_4 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k +12);

                        m2_1 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k );
                        m2_2 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k +4);
                        m2_3 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k +8);
                        m2_4 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k +12);

                        sum1 = _mm256_add_pd (sum1, _mm256_mul_pd (m1_1, m2_1)); 
                        sum2 = _mm256_add_pd (sum2, _mm256_mul_pd (m1_2, m2_2)); 
                        sum3 = _mm256_add_pd (sum3, _mm256_mul_pd (m1_3, m2_3)); 
                        sum4 = _mm256_add_pd (sum4, _mm256_mul_pd (m1_4, m2_4)); 
                    }
                 
                    for (; k < inner_dim; k++) {
                        sum_tail += mat1->data[inner_dim*i + k] * mat2_t[inner_dim*j + k];
                    }
                    _mm256_storeu_pd (sums, sum1);
                    _mm256_storeu_pd (sums+4, sum2);
                    _mm256_storeu_pd (sums+8, sum3);
                    _mm256_storeu_pd (sums+12, sum4);
                    temp[i*c + j] = sum_tail + sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5] + sums[6] + sums[7];
                    temp[i*c + j] += sums[8] + sums[9] + sums[10] + sums[11] + sums[12] + sums[13] + sums[14] + sums[15];
                    i+=1;

                    sum1 = _mm256_set1_pd(0);
                    sum2 = _mm256_set1_pd(0);
                    sum3 = _mm256_set1_pd(0);
                    sum4 = _mm256_set1_pd(0);
                    sum_tail = 0;
                    k = 0;
                    for (; k < inner_dim/unroll*unroll; k+=unroll) {
                        m1_1 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k);
                        m1_2 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k +4);
                        m1_3 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k +8);
                        m1_4 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k +12);

                        m2_1 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k );
                        m2_2 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k +4);
                        m2_3 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k +8);
                        m2_4 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k +12);

                        sum1 = _mm256_add_pd (sum1, _mm256_mul_pd (m1_1, m2_1)); 
                        sum2 = _mm256_add_pd (sum2, _mm256_mul_pd (m1_2, m2_2)); 
                        sum3 = _mm256_add_pd (sum3, _mm256_mul_pd (m1_3, m2_3)); 
                        sum4 = _mm256_add_pd (sum4, _mm256_mul_pd (m1_4, m2_4)); 
                    }

                    for (; k < inner_dim; k++) {
                        sum_tail += mat1->data[inner_dim*i + k] * mat2_t[inner_dim*j + k];
                    }
                    _mm256_storeu_pd (sums, sum1);
                    _mm256_storeu_pd (sums+4, sum2);
                    _mm256_storeu_pd (sums+8, sum3);
                    _mm256_storeu_pd (sums+12, sum4);
                    temp[i*c + j] = sum_tail + sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5] + sums[6] + sums[7];
                    temp[i*c + j] += sums[8] + sums[9] + sums[10] + sums[11] + sums[12] + sums[13] + sums[14] + sums[15];
                
                    i++;

                    sum1 = _mm256_set1_pd(0);
                    sum2 = _mm256_set1_pd(0);
                    sum3 = _mm256_set1_pd(0);
                    sum4 = _mm256_set1_pd(0);
                    sum_tail = 0;
                    k = 0;
                    for (; k < inner_dim/unroll*unroll; k+=unroll) {
                        m1_1 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k);
                        m1_2 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k +4);
                        m1_3 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k +8);
                        m1_4 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k +12);

                        m2_1 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k );
                        m2_2 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k +4);
                        m2_3 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k +8);
                        m2_4 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k +12);

                        sum1 = _mm256_add_pd (sum1, _mm256_mul_pd (m1_1, m2_1)); 
                        sum2 = _mm256_add_pd (sum2, _mm256_mul_pd (m1_2, m2_2)); 
                        sum3 = _mm256_add_pd (sum3, _mm256_mul_pd (m1_3, m2_3)); 
                        sum4 = _mm256_add_pd (sum4, _mm256_mul_pd (m1_4, m2_4)); 
                    }                  
                    for (; k < inner_dim; k++) {
                        sum_tail += mat1->data[inner_dim*i + k] * mat2_t[inner_dim*j + k];
                    }

                    _mm256_storeu_pd (sums, sum1);
                    _mm256_storeu_pd (sums+4, sum2);
                    _mm256_storeu_pd (sums+8, sum3);
                    _mm256_storeu_pd (sums+12, sum4);
                    temp[i*c + j] = sum_tail + sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5] + sums[6] + sums[7];

                    temp[i*c + j] += sums[8] + sums[9] + sums[10] + sums[11] + sums[12] + sums[13] + sums[14] + sums[15];
                
                    i-=7;
                }
            }
            #pragma omp parallel for private(sums, sum_tail,sum1, sum2, sum3, sum4, m1_1, m1_2, m1_3, m1_4, m2_1, m2_2, m2_3, m2_4) collapse(2)
            for (int i = r/8*8; i < r; i+=1) {
                for (int j = 0; j < c; j++) {
                    sum1 = _mm256_set1_pd(0);
                    sum2 = _mm256_set1_pd(0);
                    sum3 = _mm256_set1_pd(0);
                    sum4 = _mm256_set1_pd(0);
                    sum_tail = 0;
                    int k = 0;
                    for (; k < inner_dim/unroll*unroll; k+=unroll) {
                        m1_1 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k);
                        m1_2 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k +4);
                        m1_3 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k +8);
                        m1_4 = _mm256_loadu_pd(mat1->data + inner_dim * i+ k +12);

                        m2_1 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k );
                        m2_2 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k +4);
                        m2_3 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k +8);
                        m2_4 = _mm256_loadu_pd(mat2_t + inner_dim * j+ k +12);

                        sum1 = _mm256_add_pd (sum1, _mm256_mul_pd (m1_1, m2_1)); 
                        sum2 = _mm256_add_pd (sum2, _mm256_mul_pd (m1_2, m2_2)); 
                        sum3 = _mm256_add_pd (sum3, _mm256_mul_pd (m1_3, m2_3)); 
                        sum4 = _mm256_add_pd (sum4, _mm256_mul_pd (m1_4, m2_4)); 
                    }

                    for (; k < inner_dim; k++) {
                        sum_tail += mat1->data[inner_dim*i + k] * mat2_t[inner_dim*j + k];
                    }
                    _mm256_storeu_pd (sums, sum1);
                    _mm256_storeu_pd (sums+4, sum2);
                    _mm256_storeu_pd (sums+8, sum3);
                    _mm256_storeu_pd (sums+12, sum4);
                    temp[i*c + j] = sum_tail + sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5] + sums[6] + sums[7];
                    temp[i*c + j] += sums[8] + sums[9] + sums[10] + sums[11] + sums[12] + sums[13] + sums[14] + sums[15];
                }
            }
            
        }

        // memcpy(result->data, temp, size*sizeof(double));
        unroll = 8;
        int end = size/unroll*unroll;
        #pragma omp parallel for
        for (int i = 0; i < end; i+=unroll) {
            __m256d temp_vals1 = _mm256_loadu_pd(temp + i);
            __m256d temp_vals2 = _mm256_loadu_pd(temp + i+4);
            // __m256d temp_vals3 = _mm256_loadu_pd(temp + i+8);
            // __m256d temp_vals4 = _mm256_loadu_pd(temp + i+12);
            _mm256_storeu_pd (result->data + i, temp_vals1);
            _mm256_storeu_pd (result->data + i+4, temp_vals2);
            // _mm256_storeu_pd (result->data + i+8, temp_vals3);
            // _mm256_storeu_pd (result->data + i+12, temp_vals4);
        }
        for (int i = end; i < size; i++) {
            result->data[i] = temp[i];
        }
    } 
    


    // print_matrix(temp, r, c);
    free(temp);
    free(mat2_t);
    return 0;
}
int mul_matrix_naive(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.6 TODO
    double sum;
    int blocksize = 32;

    int size = result->rows*result->cols;

    int r = mat2->rows;
    int c = mat2->cols;    
    double *mat2_t = malloc(r * c * sizeof(double));
    //transpose_blocking(r, c, blocksize, mat2_t, mat2->data);
    transpose(r, c, mat2_t, mat2->data);

    // print_matrix(mat2_t, c, r);

    r = result->rows;
    c = result->cols;
    int inner_dim = mat1->cols;
    double *temp = malloc(size * sizeof(double));
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            sum = 0;
            for (int k = 0; k < inner_dim; k++) {
                sum += mat1->data[inner_dim * i + k] * mat2_t[inner_dim * j + k];
            }
            temp[i*c + j] = sum;
        }
    }

    for (int i = 0; i < size; i++) {
        result->data[i] = temp[i];
    }
    // print_matrix(temp, r, c);
    free(temp);
    free(mat2_t);
    return 0;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 * You may assume `mat` is a square matrix and `pow` is a non-negative integer.
 * Note that the matrix is in row-major order.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    int n = mat->rows;
    int size = n*n;
    if (pow == 0) {
        identity(result->data, n);
        return 0;
    } else if (pow == 1) {
        memcpy(result->data, mat->data, size*sizeof(double));
        return 0;
    } else {
        // matrix *mat_pows = malloc(sizeof(matrix));
        // mat_pows->data = malloc(size * sizeof(double));
        // memcpy(mat_pows->data, mat->data, size*sizeof(double));        
        // mat_pows->cols = n;
        // mat_pows->rows = n;
        double *backup = malloc(size * sizeof(double));
        memcpy(backup, mat->data, size*sizeof(double));

        identity(result->data, n);
        while (pow > 0) {
            if (pow & 1)
                mul_matrix(result, result, mat);
            pow = pow >> 1;
            if (pow > 0)
                mul_matrix(mat, mat, mat);
        }
        memcpy(mat->data, backup, size*sizeof(double));
        //free(mat_pows->data);
        // free(backup);
        return 0;
    }
    return 0;
}
int pow_matrix_old(matrix *result, matrix *mat, int pow) {
    int n = mat->rows;
    int size = n*n;
    if (pow == 0) {
        identity(result->data, n);
        return 0;
    } else if (pow == 1) {
        memcpy(result->data, mat->data, size*sizeof(double));
        return 0;
    } else {
        // x = mat = result
        memcpy(result->data, mat->data, size*sizeof(double));

        matrix *y = malloc(sizeof(matrix));
        y->data = malloc(size * sizeof(double));
        y->cols = n;
        y->rows = n;
        identity(y->data, n);
        int iter = 0;
        while (pow > 1) {
            if (pow & 1) {
                mul_matrix(y, result, y);
                mul_matrix(result, result, result);
                pow = (pow-1) >> 1;
            } else {
                mul_matrix(result, result, result);
                pow = pow >> 1;
            }
        }

        mul_matrix(result, result, y);

        free(y->data);
        free(y);
        return 0;
    }
    return 0;
}
void identity(double *y, int n) {
    // #pragma omp parallel for
    // for (int i = 0; i < n*n; i++) {
    //     mat[i] = 0;
    // }
    // for (int i = 0; i<n; i++) {
    //     mat[i*n+i] = 1;
    // }
    int unroll = 16;
    int size = n*n;
    int end = n*n/unroll*unroll;
    __m256d zero =  _mm256_set1_pd(0);
    // #pragma omp parallel for
    for (int i = 0; i < end; i+=unroll) {
        _mm256_storeu_pd((y + i), zero);
        _mm256_storeu_pd((y + i+4), zero);
        _mm256_storeu_pd((y + i+8), zero);
        _mm256_storeu_pd((y + i+12), zero);        
    }
    for (int i = end; i < size; i++) {
        y[i] = 0;
    }
    for (int i = 0; i<n; i++) {
        y[i*n+i] = 1;
    }
}
int pow_matrix_naive(matrix *result, matrix *mat, int pow) {
    int size = mat->rows * mat->cols;
    if (pow == 0) {
        for (int i = 0; i < size; i++) {
            if (i/result->cols == i % result->cols) {
                set(result, i/result->cols, i % result->cols, 1);
            } else {
                set(result, i/result->cols, i % result->cols, 0);
            }
        }
        return 0;
    } else if (pow == 1) {
        double val;
        for (int j = 0; j < size; j++) {
            val = get(mat, j/mat->cols, j % mat->cols);
            set(result, j/result->cols, j%result->cols, val);
        }
        return 0;
    } else {
        for (int i = 1; i < pow; i++) {
            if (i == 1) {
                mul_matrix(result, mat, mat);
            } else {
                mul_matrix(result, result, mat);
            }
        }
    }
    return 0;
}

void print_matrix(double *mat, int r, int c) {
    printf("\n");
    for (int i = 0; i < r*c; i++) {
        printf("%f ", mat[i]);
        if ((i+1) % c == 0) {
            printf("\n");
        }
    }
}

void print_m256d(__m256d d) {
    double temp[4];
    _mm256_storeu_pd (temp, d);
    printf("\n values: %f %f %f %f", temp[0], temp[1],temp[2], temp[3]);
}