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
