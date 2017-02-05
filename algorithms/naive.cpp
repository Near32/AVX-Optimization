void multCNaive(double *A, double *x, double *y, int N)
{
	int i,k;
	
	for(i=0;i<N;i++)
	{
		y[i]=0;
		for(k=0;k<N;k++)
		{
			y[i] += A[i*N+k]*x[k];
		} 
	}
	
}
