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
		
		*(py) += dty[0]+dty[1];
		
		
		for(size_t rr=r;rr--;)
		{
      *(py) += *(pa++) * *(px++);
		}
		
		py++;
	}
		
}
