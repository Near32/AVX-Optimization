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
