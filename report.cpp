#include <immintrin.h>
#include <time.h>
#include <iostream>
//#include "./RunningStats/RunningStats.h"


void multCNaive(double *A, double *x, double *y, int N)
{
	int i,j,k;
	
	for(i=0;i<N;i++)
	{
		y[i]=0;
		for(k=0;k<N;k++)
		{
			y[i] += A[i*N+k]*x[k];
		} 
	}
	
}


void multCUnrolled(double *A, double *x, double *y, int N)
{
	int d=4;
	int q = N/d;
	int r = N%d;
	double *pa = A;
	double *py = y;
	
	for(size_t i=0;i<N;i++)
	{
		pa = A+i*N;
		double *px = x;
		
		(*py)=0;
		for(size_t k=0;k<q;k++)
		{
			(*py) += *(pa) * *(px)
						+ *(pa+1) * *(px+1)
						+ *(pa+2) * *(px+2)
						+ *(pa+3) * *(px+3);
						
			px+=d;
			pa+=d;
		}
		
		for(size_t rr=r;rr--;)
		{
			(*py) += *(pa) * *(px);
			px++;
			pa++;
		}
		
		py++; 
	}
		
}


void multAVX1(double *A, double *x, double *y, int N)
{
	int d=4;
	int times = 2;
	int q = N/(d*times);
	int r = N%(d*times);
	double *pa = A;
	double *py = y;
	double *ytemp = (double*)aligned_alloc(64,d*sizeof(double));
	
	for(size_t i=0;i<N;i++)
	{
		pa = A+i*N;
		double *px = x;
		
		*(py)=0;
		
		for(size_t k=0;k<q;k++)
		{
			__m256d AA1,AA2,xx1,xx2,yy;
			AA1 = _mm256_load_pd(pa);
			xx1 = _mm256_load_pd(px);
			AA2 = _mm256_load_pd(pa+d);
			xx2 = _mm256_load_pd(px+d);
			
			yy = _mm256_hadd_pd( _mm256_mul_pd(AA1,xx1), _mm256_mul_pd(AA2,xx2) );
			_mm256_store_pd(ytemp, yy);
			
			*(py) += ytemp[0]+ytemp[1]+ytemp[2]+ytemp[3];
			
						
			px+=times*d;
			pa+=times*d;
		}
		
		for(size_t rr=r;rr--;)
		{
			(*py) += *(pa) * *(px);
			px++;
			pa++;
		}
		
		py++; 
	}
		
	free(ytemp);
}

void multAVX2(double *A, double *x, double *y, int N)
{
	int d=4;
	int q = N/d;
	int r = N%d;
	double *pa = A;
	double *py = y;
	double *px = x;
	__m256d xx;
	__m256d AA;
	__m256d ty;
		
	for(size_t i=0;i<N;i++)
	{
		
		*(py)=0;
		px = x;
			
		ty = _mm256_set1_pd( *py );
		
		for(size_t k=0;k<q;k++)
		{
			AA = _mm256_load_pd(pa);
			xx = _mm256_load_pd(px);
			
			ty = _mm256_add_pd( ty,_mm256_mul_pd(AA,xx) );
			
			pa+=d;
			px+=d;
		}
		
		double dty[4];
		
		ty = _mm256_hadd_pd( ty, ty);
		_mm256_store_pd(dty, ty);
		
		//*(py) += dty[0]+dty[1]+dty[2]+dty[3];
		*(py) += dty[0]+dty[1];
		
		
		for(size_t rr=r;rr--;)
		{
      *(py) += *(pa++) * *(px++);
		}
		
		py++;
	}
		
}


void multAVX2FMA(double *A, double *x, double *y, int N)
{
	int d=4;
	int q = N/d;
	int r = N%d;
	double *pa = A;
	double *py = y;
	double *px = x;
	__m256d xx;
	__m256d AA;
	__m256d ty;
		
	for(size_t i=0;i<N;i++)
	{
		
		*(py)=0;
		px = x;
			
		ty = _mm256_set1_pd( *py );
		
		for(size_t k=0;k<q;k++)
		{
			AA = _mm256_load_pd(pa);
			xx = _mm256_load_pd(px);
			
			//ty = _mm256_add_pd( ty,_mm256_mul_pd(AA,xx) );
			ty = _mm256_fmadd_pd( AA, xx, ty );
			
			pa+=d;
			px+=d;
		}
		
		double dty[4];
		
		ty = _mm256_hadd_pd( ty, ty);
		_mm256_store_pd(dty, ty);
		
		//*(py) += dty[0]+dty[1]+dty[2]+dty[3];
		*(py) += dty[0]+dty[1];
		
		
		for(size_t rr=r;rr--;)
		{
      *(py) += *(pa++) * *(px++);
		}
		
		py++;
	}
		
}

void multAVX(double *A, double *x, double *y, int N)
{
	int d=4;
	int q = N/d;
	int r = N%d;
	double *pa = A;
	
	double *py = y;
	
	__m256d AA,xx,yy;
	
	for(size_t i=0;i<q;i++)
	{
		double *px = x;
		AA = _mm256_load_pd(pa);
		
		yy = _mm256_setzero_pd();
		
		for(size_t k=0;k--;)
		{
			xx = _mm256_load_pd(px);
			
			//yy = _mm256_add_pd(yy, _mm256_mul_pd() );
			
			px+=d;
		}
		
		pa+=d;
		px+=d;
	}
	
	for(r=q*d;r<N;r++)
	{
		double *px = x;
		*(py) = 0;
		
		for(size_t k=0;k<N;k++)
		{ 
			*(py) += (*(pa+k)) * (*(px++));
		}
		
		py++;
		pa++;
		
	}
	
}

void print(double *x, size_t N)
{
	for(size_t i=0;i<N;i++)
	{
		std::cout << x[i];
		
		if(i!=N-1)
		{
			std::cout << " , ";
		}
		
	}
	
	std::cout << std::endl;
}
	
int main(int argc, char* argv[])
{
	srand(time(NULL));
	
	float factor = 1e6f;
	bool verbose = false;
	if(argc>1)
		verbose = (atoi(argv[1])==1?true:false);
	else
		std::cout << " USAGE :: report.output [verbose (int) 0:false / 1:true]	[N (unsigned int)]" << std::endl;
	
	float duration;
	unsigned int N=4;
	if(argc>2)
		N = atoi(argv[2]);
	else
		std::cout << " USAGE :: report.output [verbose default=0 (int) 0:false / 1:true]	[N default=4 (unsigned int)]" << std::endl;
	
	double *A = (double*)aligned_alloc(64,N*N*sizeof(double));
	double *tA = (double*)aligned_alloc(64,N*N*sizeof(double));
	double *x = (double*)aligned_alloc(64,N*sizeof(double));
	double *y = (double*)aligned_alloc(64,N*sizeof(double));
	
	
	int i,j;
	for(i=0;i<N;i++)
	{
		for(j=0;j<N;j++)
		{
			A[i*N+j] = float(rand()%10);
			tA[j*N+i] = A[i*N+j];
		}
		
		x[i] = float(rand()%10);
	}
	
	
	clock_t time = clock();
	
	multCNaive(A,x,y,N);
	
	duration = float(clock()-time)*factor/CLOCKS_PER_SEC;
	std::cout << " NAIVE C/C++ Mult : " << duration << " microseconds." << std::endl;
	if(verbose)	print(y,N);
	
	//---------------------------------------------------------------------------
	
	time = clock();
	
	multCUnrolled(A,x,y,N);
	
	
	duration = float(clock()-time)*factor/CLOCKS_PER_SEC;
	std::cout << " UNROLLED C/C++ Mult : " << duration << " microseconds." << std::endl;
	if(verbose)	print(y,N);
	
	//---------------------------------------------------------------------------
	time = clock();
	
	multAVX1(A,x,y,N);
	duration = float(clock()-time)*factor/CLOCKS_PER_SEC;
	std::cout << " AVX Mult : " << duration << " microseconds." << std::endl;
	if(verbose)	print(y,N);
	
	//---------------------------------------------------------------------------
	time = clock();
	
	multAVX2(A,x,y,N);
	duration = float(clock()-time)*factor/CLOCKS_PER_SEC;
	std::cout << " AVX2 Mult : " << duration << " microseconds." << std::endl;
	if(verbose)	print(y,N);
	
	//---------------------------------------------------------------------------
	time = clock();
	
	multAVX2FMA(A,x,y,N);
	duration = float(clock()-time)*factor/CLOCKS_PER_SEC;
	std::cout << " AVX2+FMA Mult : " << duration << " microseconds." << std::endl;
	if(verbose)	print(y,N);
	
	free(A);
	free(tA);
	free(x);
	free(y);
	
	return 0;
	
}
