#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <omp.h>

static double total_Ap_time = 0.0;
static double total_Z_time = 0.0;
static double total_sumc_time = 0.0;
static double total_dot_time = 0.0;
static double total_energy_time = 0.0;

typedef struct {
    double *a;
    double *b;
    double *F;
    double *D;
    int M, N;
    int size;
    double x_min, x_max;
    double y_min, y_max;
    double hx, hy, ieps;
    double ihx, ihy, ihx2, ihy2, hxhy;
    double delta;
} CGParams;

typedef struct {
    int iter;
    double *result;
    double *difference;
    // double **grid_error;
    double *energy_func;
} CGResults;

CGParams* create_cg_params(int M, int N) {
    int size = (M - 1)*(N - 1);

    CGParams *cg_params = (CGParams*)malloc(sizeof(CGParams));
    cg_params->M = M;
    cg_params->N = N;
    cg_params->size = size;
    cg_params->a = (double*)malloc(M*N*sizeof(double));
    cg_params->b = (double*)malloc(M*N*sizeof(double));
    cg_params->F = (double*)malloc(size*sizeof(double));
    cg_params->D = (double*)malloc(size*sizeof(double));
    return cg_params;
}

CGResults* create_cg_result(CGParams *cg_params) {
    int size = cg_params->size;

    CGResults *result = (CGResults*)malloc(sizeof(CGResults));
    result->result = (double*)malloc(size*sizeof(double));
    result->difference = (double*)malloc(size*sizeof(double));
    result->energy_func = (double*)malloc(size*sizeof(double));
    // result->grid_error = (double**)malloc(size*sizeof(double*));
    return result;
}

void set_x_params(CGParams *cg_params, double x_min, double x_max) {
    cg_params->x_min = x_min;
    cg_params->x_max = x_max;
}

void set_y_params(CGParams *cg_params, double y_min, double y_max) {
    cg_params->y_min = y_min;
    cg_params->y_max = y_max;
}

void set_grid_params(CGParams *cg_params, double delta) {
    double hx = (cg_params->x_max - cg_params->x_min)/cg_params->M;
    double hy = (cg_params->y_max - cg_params->y_min)/cg_params->N;
    double hxhy = hx*hy;
    double eps = (hx > hy) ? (hx*hx) : (hy*hy);

    cg_params->hx = hx;
    cg_params->ihx = 1.0/hx;
    cg_params->ihx2 = 1.0/(hx*hx);

    cg_params->hy = hy;
    cg_params->ihy = 1.0/hy;
    cg_params->ihy2 = 1.0/(hy*hy);

    cg_params->ieps = 1.0/eps;
    cg_params->hxhy = hxhy;
    cg_params->delta = delta;
}

void free_cg_params(CGParams *cg_params) {
    free(cg_params->a);
    free(cg_params->b);
    free(cg_params->F);
    free(cg_params->D);
    free(cg_params);
}

void free_cg_result(CGParams *cg_params, CGResults *cg_results) {
    int size = cg_params->size;
    free(cg_results->result);
    free(cg_results->difference);
    free(cg_results->energy_func);
    // for (int i = 0; i < size; i++) {
    //     free(cg_results->grid_error[i]);
    // }
    // free(cg_results->grid_error);
    free(cg_results);
}

void compute_cg_params(CGParams *cg_params) {
    int M = cg_params->M;
    int N = cg_params->N;
    double hx = cg_params->hx;
    double hy = cg_params->hy;
    double x_min = cg_params->x_min;
    double y_min = cg_params->y_min;

    double ihx = cg_params->ihx;
    double ihy = cg_params->ihy;
    double ieps = cg_params->ieps;

    double *a = cg_params->a;
    double *b = cg_params->b;
    double *F = cg_params->F;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double x1 = x_min + (i + 0.5)*hx;
            double x2 = x_min + (i + 1.5)*hx;
            double y1 = y_min + (j + 0.5)*hy;
            double y2 = y_min + (j + 1.5)*hy;
            
            int idx = i*N + j;

            /* ----- Compute F ----- */
            if ((i < M - 1) && (j < N - 1)) {
                if ((y2 - 2*x1/3 <= 2) && (y2 + 2*x2/3 <= 2)) {
                    F[idx - i] = 1;
                } else {
                    F[idx - i] = 0;
                }
            }
            /* ----- Compute F ----- */

            /* ----- Compute a ----- */
            if (x1 >= 0) {
                if (y1 >= -2*x1/3 + 2) {
                    a[idx] = ieps;
                } else if (y2 <= -2*x1/3 + 2) {
                    a[idx] = 1;
                } else {
                    a[idx] = ((y2 + 2*x1/3 - 2)*ieps + (-2*x1/3 + 2 - y1))*ihy;
                }
            } else {
                if (y1 >= 2*x1/3 + 2) {
                    a[idx] = ieps;
                } else if (y2 <= 2*x1/3 + 2) {
                    a[idx] = 1;
                } else {
                    a[idx] = ((y2 - 2*x1/3 - 2)*ieps + (2*x1/3 + 2 - y1))*ihy;
                }               
            }
            /* ----- Compute a ----- */

            /* ----- Compute b ----- */
            if ((x2 <= 3*y1/2 - 3) || (x1 >= 3 - 3*y1/2)) {
                b[idx] = ieps;
            } else if (x1 <= 3*y1/2 - 3) {
                if (x2 <= 3 - 3*y1/2) {
                    b[idx] = ((3*y1/2 - 3 - x1)*ieps + (x2 - 3*y1/2 + 3))*ihx;
                } else {
                    b[idx] = ((x2 - x1 + 3*y1 - 6)*ieps + (6 - 3*y1))*ihx;
                }
            } else {
                if (x2 <= 3 - 3*y1/2) {
                    b[idx] = 1;
                } else {
                    b[idx] = ((x2 - 3 + 3*y1/2)*ieps + (3 - 3*y1/2 - x1))*ihx;
                }
            }
            /* ----- Compute b ----- */
        }
    }    
}

void compute_cg_d(CGParams *cg_params) {
    int M = cg_params->M;
    int N = cg_params->N;

    double ihx2 = cg_params->ihx2;
    double ihy2 = cg_params->ihy2;

    double *a = cg_params->a;
    double *b = cg_params->b;
    double *D = cg_params->D;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M - 1; i++) {
        for (int j = 0; j < N - 1; j++) {
            int idx = i*N + j;
            D[idx - i] = 1.0/((a[idx + N] + a[idx])*ihx2 + (b[idx + 1] + b[idx])*ihy2);
        }
    }
}

void compute_Ap(CGParams *cg_params, double *Ap, double *p) {
    int M = cg_params->M;
    int N = cg_params->N;
    double *a = cg_params->a;
    double *b = cg_params->b;
    double ihx = cg_params->ihx;
    double ihy = cg_params->ihy;

    double start = omp_get_wtime();
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M - 1; i++) {
        for (int j = 0; j < N - 1; j++) {
            int idx = i*(N - 1) + j;

            double fx = (i == M - 2) ? (-p[idx]*ihx) : ((p[idx + N - 1] - p[idx])*ihx);
            double fy = (j == N - 2) ? (-p[idx]*ihy) : ((p[idx + 1] - p[idx])*ihy);
            double bx = (i == 0) ? ( p[idx]*ihx) : ((p[idx] - p[idx - N + 1])*ihx);
            double by = (j == 0) ? ( p[idx]*ihy) : ((p[idx] - p[idx - 1])*ihy);

            double coefa = (a[idx + i + N]*fx - a[idx + i]*bx)*ihx;
            double coefb = (b[idx + i + 1]*fy - b[idx + i]*by)*ihy;

            double lapl = -coefa*ihx - coefb*ihy;
            Ap[idx] = lapl;
        }
    }
    double end = omp_get_wtime();
    total_Ap_time += (end - start);
}

void compute_Z(CGParams *cg_params, double *z, double *r) {
    int size = cg_params->size;
    double *D = cg_params->D;

    double start = omp_get_wtime();
    #pragma omp parallel for simd
    for (int idx = 0; idx < size; idx++) {
        z[idx] = r[idx]*D[idx];
    }
    double end = omp_get_wtime();
    total_Z_time += (end - start);
}

double dot(CGParams *cg_params, double *v, double *w) {
    int size = cg_params->size;
    double hxhy = cg_params->hxhy;
    
    double dot_product = 0;
    double start = omp_get_wtime();
    #pragma omp parallel for simd reduction(+:dot_product)
    for (int i = 0; i < size; i++) {
        dot_product += v[i]*w[i];
    }
    double end = omp_get_wtime();
    total_dot_time += (end - start);
    return dot_product*hxhy;
}

double norm(CGParams *cg_params, double *v) {
    double eucl_norm = dot(cg_params, v, v);
    return sqrt(eucl_norm);
}

void array_sumc(double *v, double *w, double *sumc, int size, double coef) {
    double start = omp_get_wtime();
    #pragma omp parallel for simd
    for (int i = 0; i < size; i++) {
        sumc[i] = v[i] + w[i]*coef;
    }
    double end = omp_get_wtime();
    total_sumc_time += (end - start);
}

double compute_cg_energy(CGParams *cg_params, double *r, double *w) {
    int size  = cg_params->size;
    double *F = cg_params->F;

    double energy = 0.0;
    double start = omp_get_wtime();
    #pragma omp parallel for simd reduction(+:energy)
    for (int i = 0; i < size; i++) {
        energy += (F[i] + r[i]) * w[i];
    }
    double end = omp_get_wtime();
    total_energy_time += (end - start);
    return -0.5*energy;
}

// double* compute_cg_grid_error(CGParams *cg_params, double *p, double alpha) {
//     int size = cg_params->size;
//     double *grid_error = (double*)malloc(size*sizeof(double));
//     #pragma omp parallel for simd
//     for (int i = 0; i < size; i++) {
//         grid_error[i] = alpha*p[i];
//     }
//     return grid_error;
// }

void compute_cg_solution(CGParams *cg_params, CGResults *cg_result) {
    int size  = cg_params->size;
    double *F = cg_params->F;

    double *z  = (double*)malloc(size*sizeof(double));
    double *r  = (double*)malloc(size*sizeof(double));
    double *p  = (double*)malloc(size*sizeof(double));
    double *Ap = (double*)malloc(size*sizeof(double));

    double *w = cg_result->result;
    double *energy_func = cg_result->energy_func;
    double *difference  = cg_result->difference;
    // double **grid_error = cg_result->grid_error;

    double alpha, beta;
    double old_zr, new_zr;

    /* ----- Initial step of CG ----- */
    memcpy(r, F, size*sizeof(double));
    compute_Z(cg_params, z, r);
    memcpy(p, z, size*sizeof(double));
    old_zr = dot(cg_params, z, r);
    compute_Ap(cg_params, Ap, p); 
    alpha = old_zr/dot(cg_params, Ap, p);
    array_sumc(w, p, w, size, alpha);
    /* ----- Initial step of CG ----- */

    /* ----- CG loop ----- */
    int cg_iter;
    for (cg_iter = 0; cg_iter < size; cg_iter++) {
        array_sumc(r, Ap, r, size, -alpha);    
        compute_Z(cg_params, z, r);
        new_zr = dot(cg_params, z, r);
        beta = new_zr/old_zr;
        array_sumc(z, p, p, size, beta);
        compute_Ap(cg_params, Ap, p);
        alpha = new_zr/dot(cg_params, Ap, p);
        array_sumc(w, p, w, size, alpha);
        old_zr = new_zr;

        energy_func[cg_iter] = compute_cg_energy(cg_params, r, w);
        // grid_error[cg_iter] = compute_cg_grid_error(cg_params, p, alpha);

        double delta = fabs(alpha)*norm(cg_params, p);
        difference[cg_iter] = delta;
        if (delta < cg_params->delta) {
            printf("CG has finished (%d): delta = %.2e\r\n", cg_iter, delta);
            break;
        }
    }
    /* ----- CG loop ----- */

    cg_result->iter = cg_iter;

    free(z);
    free(r);
    free(p);
    free(Ap);
}

void write_cg_results(CGParams *cg_params, CGResults *cg_results) {
    int M = cg_params->M;
    int N = cg_params->N;
    int size = cg_params->size;

    int iter = cg_results->iter;
    double *result = cg_results->result;
    double *difference = cg_results->difference;
    double *energy_func = cg_results->energy_func;
    // double **grid_error = cg_results->grid_error;

    FILE* file1 = fopen("cg_result.csv", "w");
    FILE* file2 = fopen("cg_energy_func.csv", "w");
    FILE* file3 = fopen("cg_difference.csv", "w");
    // FILE* file4 = fopen("cg_grid_error.csv", "w");
    if ((file1 == NULL) || (file2 == NULL) || (file3 == NULL)) {
        printf("Error opening file!\n\r");
        return;
    }

    for (int i = 0; i < M - 1; i++) {
        for (int j = 0; j < N - 1; j++) {
            int idx = i*(N - 1) + j;
            if (j == N - 2) {
                fprintf(file1, "%f", result[idx]);
            } else {
                fprintf(file1, "%f,", result[idx]);
            }
        }
        fprintf(file1,"\r\n");
    }

    for (int i = 0; i < iter; i++) {
        if (i == iter - 1) {
            fprintf(file2, "%f", energy_func[i]);
            fprintf(file3, "%f", difference[i]);
        } else {
            fprintf(file2, "%f,", energy_func[i]);
            fprintf(file3, "%f,", difference[i]);
        }
    }
    
    // for (int i = 0; i < iter; i++) {
    //     for (int idx = 0; idx < size; idx++) {
    //         if (idx == size - 1) {
    //             fprintf(file4, "%f", grid_error[i][idx]);
    //         } else {
    //             fprintf(file4, "%f,", grid_error[i][idx]);
    //         }
    //     }
    //     fprintf(file4, "\n");
    // }

    fclose(file1);
    fclose(file2);
    fclose(file3);
    // fclose(file4);
}

int main() {
    /* ----- Define the domain ----- */
    int M = 400, N = 600;
    double x_min = -3.0, x_max = 3.0;
    double y_min =  0.0, y_max = 2.0;
    double delta = 1e-10;
    /* ----- Define the domain ----- */

    /* ----- Initialize CG parameters ----- */
    CGParams *cg_params = create_cg_params(M, N);
    set_x_params(cg_params, x_min, x_max);
    set_y_params(cg_params, y_min, y_max);
    set_grid_params(cg_params, delta);

    double start_params = omp_get_wtime();
    compute_cg_params(cg_params);
    compute_cg_d(cg_params);
    double end_params   = omp_get_wtime();
    /* ----- Initialize CG parameters ----- */

    /* ----- CG ----- */
    double start_cg = omp_get_wtime();
    CGResults *cg_results = create_cg_result(cg_params);
    compute_cg_solution(cg_params, cg_results);
    double end_cg   = omp_get_wtime();
    /* ----- CG ----- */

    write_cg_results(cg_params, cg_results);

    double params_time = end_params - start_params;
    double cg_time = end_cg - start_cg;
    double total_time = params_time + cg_time;
    printf("Params:\t%8.5f sec\r\n", params_time);
    printf("CG:\t%8.5f sec\r\n", cg_time);
    printf("Total:\t%8.5f sec\r\n", total_time);

    printf("compute_Ap:\t%8.5f sec\r\n", total_Ap_time);
    printf("compute_Z:\t%8.5f sec\r\n",  total_Z_time);
    printf("sumc:\t%8.5f sec\r\n",       total_sumc_time);
    printf("dot:\t%8.5f sec\r\n",        total_dot_time);
    printf("energy\t%8.5f sec\r\n",      total_energy_time);

    free_cg_result(cg_params, cg_results);
    free_cg_params(cg_params);

    return 0;
}