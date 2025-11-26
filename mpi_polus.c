#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <mpi.h>

typedef struct {
    double *a, *b, *F, *D;
    int M, N;
    int size;
    double x_min, y_min;
    double hx, hy, ieps;
    double ihx, ihy, ihx2, ihy2, hxhy;
    double delta;
    
    int rank, num_procs;            
    int global_M, global_N;       
    int global_size;
    double global_x_min, global_x_max;
    double global_y_min, global_y_max;
    
    int left, right, top, bottom;
    int has_left, has_right, has_top, has_bottom;
    double *left_ghost, *right_ghost, *top_ghost, *bottom_ghost;

    int cg_iter;
    double *result, *error, *residual, *energy_func;
    int *offsets, *sizes;
} CGParams;

CGParams* create_cg_params(int rank, int num_procs) {
    CGParams *cg_params = (CGParams*)malloc(sizeof(CGParams));
    cg_params->rank = rank;
    cg_params->num_procs = num_procs;

    if (rank == 0) {
        cg_params->offsets = (int*)malloc(num_procs*sizeof(int));
        cg_params->sizes   = (int*)malloc(num_procs*sizeof(int));
    }

    return cg_params;
}

void set_global_x_params(CGParams *cg_params, double x_min, double x_max) {
    cg_params->global_x_min = x_min;
    cg_params->global_x_max = x_max;
}

void set_global_y_params(CGParams *cg_params, double y_min, double y_max) {
    cg_params->global_y_min = y_min;
    cg_params->global_y_max = y_max;
}

void set_global_grid_params(CGParams *cg_params, int M, int N) {
    double hx = (cg_params->global_x_max - cg_params->global_x_min)/M;
    double hy = (cg_params->global_y_max - cg_params->global_y_min)/N;

    cg_params->global_M = M;
    cg_params->global_N = N;

    cg_params->global_size = M*N;

    cg_params->hx = hx;
    cg_params->hy = hy;
}

void compute_subdomain_num(CGParams *cg_params, int *proc_rows, int *proc_cols) {
    int global_M = cg_params->global_M;
    int global_N = cg_params->global_N;
    int num_procs = cg_params->num_procs;

    if ((global_M == 400) && (global_N == 600)) {
        switch (num_procs)
        {
        case 1:
            *proc_rows = 1;
            *proc_cols = 1;
            break;
        case 2:
            *proc_rows = 1;
            *proc_cols = 2;
            break;
        case 4:
            *proc_rows = 2;
            *proc_cols = 2;
            break;
        case 8:
            *proc_rows = 2;
            *proc_cols = 4;
            break;
        case 16:
            *proc_rows = 4;
            *proc_cols = 4;
            break;
        case 32:
            *proc_rows = 4;
            *proc_cols = 8;
            break;
        case 64:
            *proc_rows = 8;
            *proc_cols = 8;
            break;
        default:
            printf("Incorrect number of procs.\r\n");
            break;
        }
    } else if ((global_M == 800) && (global_N == 1200)) {
        switch (num_procs)
        {
        case 1:
            *proc_rows = 1;
            *proc_cols = 1;
            break;
        case 2:
            *proc_rows = 1;
            *proc_cols = 2;
            break;
        case 4:
            *proc_rows = 2;
            *proc_cols = 2;
            break;
        case 8:
            *proc_rows = 2;
            *proc_cols = 4;
            break;
        case 16:
            *proc_rows = 4;
            *proc_cols = 4;
            break;
        case 32:
            *proc_rows = 4;
            *proc_cols = 8;
            break;
        case 64:
            *proc_rows = 8;
            *proc_cols = 8;
            break;
        default:
            printf("Incorrect number of procs.\r\n");
            break;
        }

    } else {
        printf("Too bad you have chosen a lazy approach.\r\n");
    }
    /* Lazy approach, but good enough for the task */
}

void set_subdomain_size(CGParams *cg_params) {
    int global_M = cg_params->global_M;
    int global_N = cg_params->global_N;
    int rank = cg_params->rank;
    int num_procs = cg_params->num_procs;

    int proc_rows;
    int proc_cols;
    compute_subdomain_num(cg_params, &proc_rows, &proc_cols);

    int M = global_M/proc_rows;
    int N = global_N/proc_cols;
    int size = M*N;

    cg_params->M = M;
    cg_params->N = N;
    cg_params->size = size;

    int proc_row = rank/proc_cols;
    int proc_col = rank%proc_cols;

    double x_min = cg_params->global_x_min + proc_row*M*cg_params->hx;
    double y_min = cg_params->global_y_min + proc_col*N*cg_params->hy;
    cg_params->x_min = x_min;
    cg_params->y_min = y_min;
    
    cg_params->left   = (proc_col == 0) ? MPI_PROC_NULL : rank - 1;
    cg_params->right  = (proc_col == proc_cols - 1) ? MPI_PROC_NULL : rank + 1;

    cg_params->top    = (proc_row == 0) ? MPI_PROC_NULL : rank - proc_cols;
    cg_params->bottom = (proc_row == proc_rows - 1) ? MPI_PROC_NULL : rank + proc_cols;
    
    cg_params->has_left   = (cg_params->left   != MPI_PROC_NULL);
    cg_params->has_right  = (cg_params->right  != MPI_PROC_NULL);
    cg_params->has_top    = (cg_params->top    != MPI_PROC_NULL);
    cg_params->has_bottom = (cg_params->bottom != MPI_PROC_NULL);

    cg_params->a = (double*)malloc((size + N)*sizeof(double));
    cg_params->b = (double*)malloc((size + M)*sizeof(double));
    cg_params->F = (double*)malloc(size*sizeof(double));
    cg_params->D = (double*)malloc(size*sizeof(double));

    cg_params->left_ghost   = (double*)malloc(M*sizeof(double));
    cg_params->right_ghost  = (double*)malloc(M*sizeof(double));
    cg_params->top_ghost    = (double*)malloc(N*sizeof(double));
    cg_params->bottom_ghost = (double*)malloc(N*sizeof(double));

    if (rank == 0) {
        int global_size = cg_params->global_size;
        cg_params->result      = (double*)malloc(global_size*sizeof(double));
        cg_params->error       = (double*)malloc(global_size*sizeof(double));
        cg_params->energy_func = (double*)malloc(global_size*sizeof(double));
        cg_params->residual    = (double*)malloc(global_size*sizeof(double));
    }
}

void set_cg_constants(CGParams *cg_params, double delta) {
    double hx = cg_params->hx;
    double hy = cg_params->hy;
    double eps = (hx > hy) ? (hx*hx) : (hy*hy);

    cg_params->hxhy = hx*hy;
    cg_params->ihx = 1.0/hx;
    cg_params->ihy = 1.0/hy;
    cg_params->ihx2 = 1.0/(hx*hx);
    cg_params->ihy2 = 1.0/(hy*hy);
    cg_params->ieps = 1.0/eps;
    cg_params->delta = delta;
}

void free_cg_results(CGParams *cg_params) {
    free(cg_params->result);
    free(cg_params->error);
    free(cg_params->energy_func);
    free(cg_params->residual);
    free(cg_params->offsets);
    free(cg_params->sizes);
}

void free_cg_params(CGParams *cg_params) {
    free(cg_params->a);
    free(cg_params->b);
    free(cg_params->F);
    free(cg_params->D);
    free(cg_params->left_ghost);
    free(cg_params->right_ghost);
    free(cg_params->top_ghost);
    free(cg_params->bottom_ghost);
    free(cg_params);
}

void compute_cg_params(CGParams *cg_params) {
    int M = cg_params->M;
    int N = cg_params->N;
    double hx = cg_params->hx;
    double hy = cg_params->hy;
    double x_min = cg_params->x_min + hx/2;
    double y_min = cg_params->y_min + hy/2;

    double ihx = cg_params->ihx;
    double ihy = cg_params->ihy;
    double ieps = cg_params->ieps;

    double *a = cg_params->a;
    double *b = cg_params->b;
    double *F = cg_params->F;

    /* ----- Compute a ----- */
    for (int i = 0; i < M + 1; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i*N + j;
            double x1 = x_min + (i - 0.5)*hx;
            double y1 = y_min + (j - 0.5)*hy;
            double y2 = y_min + (j + 0.5)*hy;

            if ((x1 <= -3.0) || (x1 >= 3.0)) {
                a[idx] = ieps;
            } else {
                if (x1 <= 0) {
                    if (y1 <= 0) {
                        if (y2 <= 0) {
                            a[idx] = ieps;
                        } else if (y2 <= 2*x1/3 + 2) {
                            a[idx] = ihy*(y2 - y1*ieps);
                        } else {
                            a[idx] = ihy*((y2 - 2*x1/3 - 2 - y1)*ieps + 2*x1/3 + 2);
                        }
                    } else if (y1 <= 2*x1/3 + 2) {
                        if (y2 <= 2*x1/3 + 2) {
                            a[idx] = 1;
                        } else {
                            a[idx] = ihy*((y2 - 2*x1/3 - 2)*ieps + 2*x1/3 + 2 - y1);
                        }
                    } else {
                        a[idx] = ieps;
                    }
                } else {
                    if (y1 <= 0) {
                        if (y2 <= 0) {
                            a[idx] = ieps;
                        } else if (y2 <= -2*x1/3 + 2) {
                            a[idx] = ihy*(y2 - y1*ieps);
                        } else {
                            a[idx] = ihy*((y2 + 2*x1/3 - 2 - y1)*ieps - 2*x1/3 + 2);
                        }
                    } else if (y1 <= -2*x1/3 + 2) {
                        if (y2 <= -2*x1/3 + 2) {
                            a[idx] = 1;
                        } else {
                            a[idx] = ihy*((y2 + 2*x1/3 - 2)*ieps - 2*x1/3 + 2 - y1);
                        }
                    } else {
                        a[idx] = ieps;
                    }
                }
            }
        }
    }
    /* ----- Compute a ----- */

    /* ----- Compute b ----- */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N + 1; j++) {
            int idx = i*(N + 1) + j;
            double x1 = x_min + (i - 0.5)*hx;
            double x2 = x_min + (i + 0.5)*hx;
            double y1 = y_min + (j - 0.5)*hy;

            if ((y1 >= 2.0) || (y1 <= 0.0)) {
                b[idx] = ieps;
            } else {
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
            }
        }
    }
    /* ----- Compute b ----- */

    /* ----- Compute F ----- */
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i*N + j;
            double x1 = x_min + (i - 0.5)*hx;
            double x2 = x_min + (i + 0.5)*hx;
            double y1 = y_min + (j - 0.5)*hy;
            double y2 = y_min + (j + 0.5)*hy;

            if ((y2 - 2*x1/3 <= 2) && (y2 + 2*x2/3 <= 2) && (y1 >= 0)) {
                F[idx] = 1;
            } else {
                F[idx] = 0;
            }
        }
    }
    /* ----- Compute F ----- */
}

void compute_cg_d(CGParams *cg_params) {
    int M = cg_params->M;
    int N = cg_params->N;

    double ihx2 = cg_params->ihx2;
    double ihy2 = cg_params->ihy2;

    double *a = cg_params->a;
    double *b = cg_params->b;
    double *D = cg_params->D;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i*N + j;
            D[idx] = 1.0/((a[idx + N] + a[idx])*ihx2 + (b[idx + i + 1] + b[idx + i])*ihy2);
        }
    }
}

void exchange_ghost_cells(CGParams *cg_params, double *local_data) {
    enum {TAG_LEFT = 1, TAG_RIGHT, TAG_TOP, TAG_BOTTOM};
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Request requests[8];
    int request_count = 0;

    int M = cg_params->M;
    int N = cg_params->N;

    int left   = cg_params->left;
    int right  = cg_params->right;
    int top    = cg_params->top;
    int bottom = cg_params->bottom;

    double *send_left   = NULL;
    double *recv_left   = NULL;
    double *send_right  = NULL;
    double *recv_right  = NULL;
    double *send_top    = NULL;
    double *recv_top    = NULL;
    double *send_bottom = NULL;
    double *recv_bottom = NULL;

    /* ----- Left exchange ----- */
    if (cg_params->has_left) {
        send_left = (double*)malloc(M*sizeof(double));
        recv_left = (double*)malloc(M*sizeof(double));
        for (int i = 0; i < M; i++) {
            send_left[i] = local_data[i*N];
        }
        MPI_Isend(send_left, M, MPI_DOUBLE, left, TAG_RIGHT, comm, &requests[request_count++]);
        MPI_Irecv(recv_left, M, MPI_DOUBLE, left, TAG_LEFT, comm, &requests[request_count++]);
    }
    /* ----- Left exchange ----- */

    /* ----- Right exchange ----- */
    if (cg_params->has_right) {
        send_right = (double*)malloc(M*sizeof(double));
        recv_right = (double*)malloc(M*sizeof(double));
        for (int i = 0; i < M; i++) {
            send_right[i] = local_data[i*N + N - 1];
        }
        MPI_Isend(send_right, M, MPI_DOUBLE, right, TAG_LEFT, comm, &requests[request_count++]);
        MPI_Irecv(recv_right, M, MPI_DOUBLE, right, TAG_RIGHT, comm, &requests[request_count++]);
    }
    /* ----- Right exchange ----- */

    /* ----- Top exchange ----- */
    if (cg_params->has_top) {
        send_top = (double*)malloc(N*sizeof(double));
        recv_top = (double*)malloc(N*sizeof(double));
        for (int j = 0; j < N; j++) {
            send_top[j] = local_data[j];
        }
        MPI_Isend(send_top, N, MPI_DOUBLE, top, TAG_BOTTOM, comm, &requests[request_count++]);
        MPI_Irecv(recv_top, N, MPI_DOUBLE, top, TAG_TOP, comm, &requests[request_count++]);
    }
    /* ----- Top exchange ----- */

    /* ----- Bottom exchange ----- */
    if (cg_params->has_bottom) {
        send_bottom = (double*)malloc(N*sizeof(double));
        recv_bottom = (double*)malloc(N*sizeof(double));
        for (int j = 0; j < N; j++) {
            send_bottom[j] = local_data[(M - 1)*N + j];
        }
        MPI_Isend(send_bottom, N, MPI_DOUBLE, bottom, TAG_TOP, comm, &requests[request_count++]);
        MPI_Irecv(recv_bottom, N, MPI_DOUBLE, bottom, TAG_BOTTOM, comm, &requests[request_count++]);
    }
    /* ----- Bottom exchange ----- */

    MPI_Waitall(request_count, requests, MPI_STATUS_IGNORE);

    /* ----- Store recieved ghost cells ----- */
    if (cg_params->has_left) {
        memcpy(cg_params->left_ghost, recv_left, M*sizeof(double));
    }
    if (cg_params->has_right) {
        memcpy(cg_params->right_ghost, recv_right, M*sizeof(double));
    }
    if (cg_params->has_top) {
        memcpy(cg_params->top_ghost, recv_top, N*sizeof(double));
    }
    if (cg_params->has_bottom) {
        memcpy(cg_params->bottom_ghost, recv_bottom, N*sizeof(double));
    }
    /* ----- Store recieved ghost cells ----- */

    free(send_left);
    free(recv_left);
    free(send_right);
    free(recv_right);
    free(send_top);
    free(recv_top);
    free(send_bottom);
    free(recv_bottom);
}

void compute_Ap(CGParams *cg_params, double *Ap, double *p) {
    int M = cg_params->M;
    int N = cg_params->N;
    double *a = cg_params->a;
    double *b = cg_params->b;
    double ihx = cg_params->ihx;
    double ihy = cg_params->ihy;

    double *left_ghost   = cg_params->left_ghost;
    double *right_ghost  = cg_params->right_ghost;
    double *top_ghost    = cg_params->top_ghost;
    double *bottom_ghost = cg_params->bottom_ghost;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i*N + j;

            /* ----- forward x ----- */
            double fx;
            if (i == M - 1) {
                if (cg_params->has_bottom) {
                    fx = (bottom_ghost[j] - p[idx])*ihx;
                } else {
                    fx = -p[idx]*ihx;
                }
            } else {
                fx = (p[idx + N] - p[idx])*ihx;
            }
            /* ----- forward x ----- */

            /* ----- forward y ----- */
            double fy;
            if (j == N - 1) {
                if (cg_params->has_right) {
                    fy = (right_ghost[i] - p[idx])*ihy;
                } else {
                    fy = -p[idx]*ihy;
                }
            } else {
                fy = (p[idx + 1] - p[idx])*ihy;
            }
            /* ----- forward y ----- */

            /* ----- backward x ----- */
            double bx;
            if (i == 0) {
                if (cg_params->has_top) {
                    bx = (p[idx] - top_ghost[j])*ihx;
                } else {
                    bx = p[idx]*ihx;
                }
            } else {
                bx = (p[idx] - p[idx - N])*ihx;
            }
            /* ----- backward x ----- */
            
            /* ----- backward y ----- */
            double by;
            if (j == 0) {
                if (cg_params->has_left) {
                    by = (p[idx] - left_ghost[i])*ihy;
                } else {
                    by = p[idx]*ihy;
                }
            } else {
                by = (p[idx] - p[idx - 1])*ihy;
            }
            /* ----- backward y ----- */

            double coefa = (a[idx + N]*fx - a[idx]*bx)*ihx;
            double coefb = (b[idx + i + 1]*fy - b[idx + i]*by)*ihy;

            double lapl = -coefa*ihx - coefb*ihy;
            Ap[idx] = lapl;
        }
    }
}

void compute_Z(CGParams *cg_params, double *z, double *r) {
    int size = cg_params->size;
    double *D = cg_params->D;

    for (int i = 0; i < size; i++) {
        z[i] = r[i]*D[i];
    }
}

double dot(CGParams *cg_params, double *v, double *w) {
    int size = cg_params->size;
    double hxhy = cg_params->hxhy;
    
    double dot_product = 0;
    for (int i = 0; i < size; i++) {
        dot_product += v[i]*w[i];
    }
    return dot_product*hxhy;
}

double norm(CGParams *cg_params, double *v) {
    double eucl_norm = dot(cg_params, v, v);
    return sqrt(eucl_norm);
}

void array_sumc(double *v, double *w, double *sumc, int size, double coef) {
    for (int i = 0; i < size; i++) {
        sumc[i] = v[i] + w[i]*coef;
    }
}

double compute_cg_energy(CGParams *cg_params, double *r, double *w) {
    int size  = cg_params->size;
    double *F = cg_params->F;

    double energy = 0.0;
    for (int i = 0; i < size; i++) {
        energy += (F[i] + r[i]) * w[i];
    }
    double global_energy;
    MPI_Reduce(&energy, &global_energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    return -0.5*global_energy;
}

void compute_cg_solution(CGParams *cg_params) {
    int size  = cg_params->size;
    int global_size = cg_params->global_size;
    double *F = cg_params->F;

    double *w  = (double*)malloc(size*sizeof(double));
    double *z  = (double*)malloc(size*sizeof(double));
    double *r  = (double*)malloc(size*sizeof(double));
    double *p  = (double*)malloc(size*sizeof(double));
    double *Ap = (double*)malloc(size*sizeof(double));

    double alpha, beta, delta;
    double old_zr, new_zr, denom;

    double global_delta;
    double global_old_zr, global_new_zr, global_denom;

    /* ----- Initial step of CG ----- */
    memcpy(r, F, size*sizeof(double));
    compute_Z(cg_params, z, r);
    memcpy(p, z, size*sizeof(double));

    old_zr = dot(cg_params, z, r);
    MPI_Allreduce(&old_zr, &global_old_zr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    old_zr = global_old_zr;

    // printf("da-da?\r\n");
    exchange_ghost_cells(cg_params, p);
    // printf("ale?\r\n");
    compute_Ap(cg_params, Ap, p); 

    denom = dot(cg_params, Ap, p);
    MPI_Allreduce(&denom, &global_denom, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    denom = global_denom;

    alpha = old_zr/denom;
    array_sumc(w, p, w, size, alpha);
    /* ----- Initial step of CG ----- */

    /* ----- CG loop ----- */
    int cg_iter;
    for (cg_iter = 0; cg_iter < global_size; cg_iter++) {
        array_sumc(r, Ap, r, size, -alpha);    
        compute_Z(cg_params, z, r);

        new_zr = dot(cg_params, z, r);
        MPI_Allreduce(&new_zr, &global_new_zr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        new_zr = global_new_zr;

        beta = new_zr/old_zr;
        array_sumc(z, p, p, size, beta);
        exchange_ghost_cells(cg_params, p);
        compute_Ap(cg_params, Ap, p);

        denom = dot(cg_params, Ap, p);        
        MPI_Allreduce(&denom, &global_denom, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        denom = global_denom;

        alpha = new_zr/denom;
        array_sumc(w, p, w, size, alpha);
        old_zr = new_zr;

        if (cg_params->rank == 0) {
            cg_params->energy_func[cg_iter] = compute_cg_energy(cg_params, r, w);
        } else {
            compute_cg_energy(cg_params, r, w);
        }

        delta = dot(cg_params, p, p);
        MPI_Allreduce(&delta, &global_delta, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        delta = fabs(alpha)*sqrt(global_delta);
        
        if (cg_params->rank == 0) {
            cg_params->error[cg_iter] = delta;
            // difference[cg_iter] = norm(cg_params, r); // невязка
        }
        if (delta < cg_params->delta) {
            if (cg_params->rank == 0) {
                cg_params->cg_iter = cg_iter;
                printf("CG has finished (%d): delta = %.2e\r\n", cg_iter, delta);
            }
            break;
        }
    }
    /* ----- CG loop ----- */

    if (cg_params->rank == 0) {
        MPI_Gather(w, size, MPI_DOUBLE, cg_params->result, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gather(w, size, MPI_DOUBLE, NULL, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    free(w);
    free(z);
    free(r);
    free(p);
    free(Ap);
}

void write_cg_results(CGParams *cg_params) {
    int M = cg_params->global_M;
    int N = cg_params->global_N;
    int size = cg_params->global_size;

    int iter = cg_params->cg_iter;
    double *result = cg_params->result;
    double *error = cg_params->error;
    double *energy_func = cg_params->energy_func;

    FILE* file1 = fopen("cg_result.csv", "w");
    FILE* file2 = fopen("cg_energy_func.csv", "w");
    FILE* file3 = fopen("cg_error.csv", "w");
    if ((file1 == NULL) || (file2 == NULL) || (file3 == NULL)) {
        printf("Error opening file!\n\r");
        return;
    }

    for (int i = 0; i < cg_params->global_size - 1; i++) {
        fprintf(file1, "%f,", result[i]);
    }
    fprintf(file1, "%f", result[cg_params->global_size - 1]);

    for (int i = 0; i < iter; i++) {
        if (i == iter - 1) {
            fprintf(file2, "%f", energy_func[i]);
            fprintf(file3, "%f", error[i]);
        } else {
            fprintf(file2, "%f,", energy_func[i]);
            fprintf(file3, "%f,", error[i]);
        }
    }

    fclose(file1);
    fclose(file2);
    fclose(file3);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ----- Define the domain ----- */
    int M = 800, N = 1200;
    double x_min = -3.1, x_max = 3.1;
    double y_min = -0.1, y_max = 2.1;
    double delta = 1e-10;
    /* ----- Define the domain ----- */

    /* ----- Initialize CG parameters ----- */
    CGParams *cg_params = create_cg_params(rank, size);
    set_global_x_params(cg_params, x_min, x_max);
    set_global_y_params(cg_params, y_min, y_max);
    set_global_grid_params(cg_params, M, N);
    set_subdomain_size(cg_params);
    set_cg_constants(cg_params, delta);
    /* ----- Initialize CG parameters ----- */

    /* ----- Compute CG coefficients ----- */
    double params_start = MPI_Wtime();
    compute_cg_params(cg_params);
    compute_cg_d(cg_params);
    double params_end = MPI_Wtime();
    double params_time = params_end - params_start;
    double max_params_time;
    MPI_Reduce(&params_time, &max_params_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    /* ----- Compute CG coefficients ----- */

    /* ----- CG ----- */
    MPI_Barrier(MPI_COMM_WORLD);
    double solution_start = MPI_Wtime();
    compute_cg_solution(cg_params);
    double solution_end = MPI_Wtime();
    double solution_time = solution_end - solution_start;
    double max_solution_time;
    MPI_Reduce(&solution_time, &max_solution_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    /* ----- CG ----- */

    if (cg_params->rank == 0) {
        write_cg_results(cg_params);
        free_cg_results(cg_params);

        printf("CG_Params coefficients time is %f seconds\r\n", max_params_time);
        printf("CG_Params solution time is %f seconds\r\n", max_solution_time);
        printf("CG_Params total time is %f seconds\r\n", max_params_time + max_solution_time);
    }

    free_cg_params(cg_params);
    MPI_Finalize();
    return 0;
}