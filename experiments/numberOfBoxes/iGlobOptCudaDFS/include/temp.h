


__global__ void globOptCUDA(double *inBox, const int inRank, int *workLen, double *min, const double inRec, const double inEps, int *workCounts)
{
	__shared__ double min_s[BLOCK_SIZE];
	__shared__ int workLen_s[BLOCK_SIZE];
	__shared__ int workLen_s_temp[BLOCK_SIZE];
	__shared__ int count[BLOCK_SIZE];
	
	double minRec = inRec;
	int i, j, k, bInd, hInd, bInd2, n;
	double curEps, h;
	
	int threadId = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	
	workLen_s[threadId] = workLen[threadId];
	min_s[threadId] = minRec;
	
	count[threadId] = 0;
	
	int wl;
	
	int half;
	
	
	double *temp = (double *) malloc(500*(2*inRank+3)*sizeof(double));
	
	
	__syncthreads();	
			
		
	n = 0;
	if(threadId == 0)
	{
		for(i = 0; i < 1024; i++)
		{
			if(workLen_s[i] == 0)
			{
				for(j = 0; j < 1024; j++)
				{
					if(workLen_s[j] > 2)
					{
						half = workLen_s[j]/2;
						workLen_s[j] -= half;
							memcpy(inBox + i*1024*(2*inRank+3), inBox + j*1024*(2*inRank+3) + (workLen_s[j])*(2*inRank+3), sizeof(double)*(2*inRank+3)*half);
							workLen_s[i] += half;
							break;
					}
				}
			}
			
			//if(minRec > min_s[blockIdx.x * 1024 + i])
			//{
				//minRec = min_s[blockIdx.x * 1024 + i];
			//}
		}
		
			
		wl = workLen_s[0];
		for(i = 0; i < 1024; i++)
		{
			if(wl < workLen_s[i]) wl = workLen_s[i];
		}
		if(wl == 0) return;
			
	}	
			
	//min_s[threadId] = minRec;	
			
		
	
	
	__syncthreads();
	
	while(workLen_s[threadId] < 1024 && count[threadId] < 100000)
	{
		if(workLen_s[threadId] > 0)
		{
			
			bInd = threadId*1024*(2*inRank+3) + (workLen_s[threadId] - 1)*(2*inRank+3);
			fnCalcFunLimitsStyblinski_CUDA(inBox + bInd, inRank);
			
			if(min_s[threadId] > inBox[bInd + 2*inRank + 2])
			{
				min_s[threadId] = inBox[bInd + 2*inRank + 2];
			}

			curEps = min_s[threadId] - inBox[bInd + 2*inRank];
			//curEps = curEps > 0 ? curEps : -curEps;	
			
			
			if(min_s[threadId] - inBox[bInd + 2*inRank] < inEps)
			{
				--workLen_s[threadId];
				n++;
			}
			else
			{
				
				/*
				if(workLen_s[threadId] > 0 && workLen_s[threadId] < 250)
				{
					for(k = workLen_s[threadId] -1; k >= 0; k--)
					{
						bInd2 = threadId*1024*(2*inRank+3) + k*(2*inRank+3);
						hInd = 0;
						h = inBox[bInd2 + 1] - inBox[bInd2];
						for(i = 0; i < inRank; i++)
						{
							if( h < inBox[bInd2 + i*2 + 1] - inBox[bInd2 + i*2]) 
							{
								h = inBox[bInd2 + i*2 + 1] - inBox[bInd2 + i*2];
								hInd = i;
							}
						}
						h = h/2.0;
						for(i = 0; i < 2; i++)
						{
							for(j = 0; j < inRank; j++)
							{
								if(j == hInd) 
								{
									inBox[bInd2 + (k*2 + i)*(2*inRank+3) + j*2] = inBox[bInd2 + j*2] + h*i;
									inBox[ bInd2 + (k*2 + i)*(2*inRank+3) + j*2 + 1] = inBox[bInd2 + j*2] + h*(i+1);
								}
								else
								{
									inBox[bInd2 + (k*2 + i)*(2*inRank+3) + j*2] = inBox[bInd2 + j*2];
									inBox[bInd2 + (k*2 + i)*(2*inRank+3) + j*2 + 1] = inBox[bInd2 + j*2 + 1];
								}
							}
							fnCalcFunLimitsStyblinski_CUDA(inBox + bInd2 +  (k*2 + i)*(2*inRank+3), inRank);
							if(min_s[threadId] > inBox[bInd2 + (k*2 + i)*(2*inRank+3) + 2*inRank + 2])
							{
								min_s[threadId] = inBox[bInd2 + (k*2 + i)*(2*inRank+3) + 2*inRank + 2];
							}
						}
					}
					
					//n = 0;
					//for(i = 0; i < 2*workLen_s[threadId]; i++)
					//{
					//	if(min_s[threadId] - temp[i*(2*inRank+3) + 2*inRank] > inEps)
					//	{
					//		memcpy(inBox + threadId*1024*(2*inRank+3) + n*(2*inRank+3),temp + i*(2*inRank+3),sizeof(double)*(2*inRank+3));
					//		++n;
					//	}
					//}
					//workLen_s[threadId] = n;
					
					
					workLen_s[threadId] *= 2;
					
					//if(n>3) break;;
	/*				
					hInd = 0;
					h = inBox[bInd + 1] - inBox[bInd];
					for(i = 0; i < inRank; i++)
					{
						if( h < inBox[bInd + i*2 + 1] - inBox[bInd + i*2]) 
						{
							h = inBox[bInd + i*2 + 1] - inBox[bInd + i*2];
							hInd = i;
						}
					}
					h = h/100.0;
					for(i = 0; i < 100; i++)
					{
						for(j = 0; j < inRank; j++)
						{
							if(j == hInd) 
							{
								temp[i*(2*inRank+3) + j*2] = inBox[bInd + j*2] + h*i;
								temp[i*(2*inRank+3) + j*2 + 1] = inBox[bInd + j*2] + h*(i+1);
							}
							else
							{
								temp[i*(2*inRank+3) + j*2] = inBox[bInd + j*2];
								temp[i*(2*inRank+3) + j*2 + 1] = inBox[bInd + j*2 + 1];
							}
						}
						fnCalcFunLimitsRozenbroke_CUDA(temp + i*(2*inRank+3), inRank);
						if(min_s[threadId] > temp[i*(2*inRank+3) + 2*inRank + 2])
						{
							min_s[threadId] = temp[i*(2*inRank+3) + 2*inRank + 2];
						}
					}
					
					for(i = 0; i < 100; i++)
					{
						if(min_s[threadId] - inEps > temp[i*(2*inRank+3) + 2*inRank])
						{
							memcpy(inBox + bInd + i*(2*inRank+3),temp + i*(2*inRank+3),sizeof(double)*(2*inRank+3));
							++workLen_s[threadId];
						}
					}
					*/
					
					
				//}
							
				
				
				if(workLen_s[threadId] > 0)
				{
					bInd = threadId*1024*(2*inRank+3) + (workLen_s[threadId] - 1)*(2*inRank+3);
					
					hInd = 0;
					h = inBox[bInd + 1] - inBox[bInd];
					for(i = 0; i < inRank; i++)
					{
						if( h < inBox[bInd + i*2 + 1] - inBox[bInd + i*2]) 
						{
							h = inBox[bInd + i*2 + 1] - inBox[bInd + i*2];
							hInd = i;
						}
					}
					for(i = 0; i < inRank; i++)
					{
						if(i == hInd) 
						{
							inBox[bInd + i*2 + 1] = inBox[bInd + i*2] + h/2.0;
							inBox[bInd + 2*inRank + 3 + i*2] = inBox[bInd + i*2] + h/2.0;
							inBox[bInd + 2*inRank + 3 + i*2 + 1] = inBox[bInd + i*2] + h;
						}
						else
						{
							inBox[bInd + 2*inRank + 3 + i*2] = inBox[bInd + i*2];
							inBox[bInd + 2*inRank + 3 + i*2 + 1] = inBox[bInd + i*2 + 1];
						}
					}
					++workLen_s[threadId];
				}
			}
			
		}

		/*
		__syncthreads();
		
		if(threadId == 0)// && (count[threadId]+1) % 10 == 0)
		{
			for(i = 0; i < 1024; i++)
			{
				if(minRec > min_s[blockIdx.x * 1024 + i])
				{
					minRec = min_s[blockIdx.x * 1024 + i];
				}
			}
		}
		
		__syncthreads();
		
		min_s[threadId] = minRec;	
		
		*/
	
		
		/*
		workLen_s_temp[threadId] = workLen[threadId];
		
		__syncthreads();
		
			
		
			
		if(workLen_s[threadId] == 0)
		{
			for(i = 0; i < 1024; i++)
			{
				if(workLen_s_temp[i] > 6 && workLen_s_temp[threadId] == 0)
				{
					atomicAdd(workLen_s_temp + i, -3);
					memcpy(inBox + bInd, inBox + i*1024*(2*inRank+3) + (workLen_s_temp[i] - 1)*(2*inRank+3), sizeof(double)*(2*inRank+3)*3);
					workLen_s_temp[threadId] += 3;
					break;
				}
			}
		}
			
			//workLen[threadId] = workLen_s_temp[threadId];
		__syncthreads();
			
		workLen_s[threadId] = workLen_s_temp[threadId];
			
			*/		
			
		//__syncthreads();	
			
		/*
		if(threadId == 0 && (count[threadId]+1) % 1000 == 0)
		{
			for(i = 0; i < 1024; i++)
			{
				if(workLen_s[i] == 0)
				{
					for(j = 0; j < 1024; j++)
					{
						if(workLen_s[j] > 2)
						{
							half = workLen_s[j]/2;
							workLen_s[j] -= half;
							memcpy(inBox + i*1024*(2*inRank+3), inBox + j*1024*(2*inRank+3) + (workLen_s[j])*(2*inRank+3), sizeof(double)*(2*inRank+3)*half);
							workLen_s[i] += half;
							break;
						}
					}
				}
			}
			
			wl = workLen_s[0];
			for(i = 0; i < 1024; i++)
			{
				if(wl < workLen_s[i]) wl = workLen_s[i];
			}
			if(wl == 0) break;
			
		}	
			
		
			
			
			
		
		__syncthreads();
		
		
		*/
		
		++count[threadId];

		
	}
	
	workLen[threadId] = workLen_s[threadId];
	min[threadId] = min_s[threadId];
	workCounts[threadId]+=n;
	
}
