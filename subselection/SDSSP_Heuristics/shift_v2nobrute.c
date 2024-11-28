//Taken from MW's dem_main.c, then adapted to avoid making a mess in the original file
//Input: file name
//In file, first line has to be dim,npoints,kpoints





// expects each point on its own line with real positions.
// expects first line to be "dim npoints reals"--from thiemard

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <time.h>

int comparedim;
bool curr_fill,fill,temp_fill;
double *best_bord, *curr_bord;
double **optiset;
int *is_in;
double epsilon;
double globallower;
double glob_bound;
int debugg;

clock_t end, start;		/* timing variables		*/
double cput;

//Split this file into main and shift parts

int cmpdbl(const void *a, const void *b)
{
  if ((*(double *)a) < (*(double *)b))
    return -1;
  else if ((*(double *) a) == (*(double *)b))
    return 0;
  return 1;
}

int cmpkeyk(const void *pt1, const void *pt2)
{
  double a=(*(double **)pt1)[comparedim], b=(*(double **)pt2)[comparedim];
  if (a<b)
    return -1;
  else if (a>b)
    return 1;
  return 0;
}

void usage()
{
  fprintf(stderr, "Usage: dem_discr [dim npoints] [file]\n\nIf file not present, read from stdin. If dim, npoints not present, \nassume header '%%dim %%npoints reals' (e.g. '2 100 reals') in file.\n");
}












/////////////////////////////////////////////////////////






double oydiscr_cell(int npoints, int dim, int rempoints,
		    double **forced, int nforced, 
		    double *lowerleft, double *upperright)
{
  double discr, maxdiscr, coordlist[dim][nforced];
  int indexes[dim];
  int i,j,k,h, status, dimension;
  double biggest[dim][nforced+1], smallest[dim][nforced+1];
  
  /*if (upperright[0]>0.999 && upperright[1]>0.999 && upperright[2]>0.999 && upperright[3]>0.999 && upperright[4]>0.999)
	  debugg-=1;*/
  //double big_bord[dim][nforced+1][dim], small_bord[dim][nforced+1][dim]; // Could do it with less memory but this will do for the moment. Need double because of upperright and unknown box choices
  double ***big_bord;
  double ***small_bord;
  big_bord=malloc(dim*sizeof(double**));
  small_bord=malloc(dim*sizeof(double**));
  for (i=0;i<dim;i++){
	  big_bord[i]=malloc((nforced+1)*sizeof(double*));
	  small_bord[i]=malloc((nforced+1)*sizeof(double*));
	  for (j=0;j<(nforced+1);j++){
		  big_bord[i][j]=malloc(dim*sizeof(double));
		  small_bord[i][j]=malloc(dim*sizeof(double));
		  for (k=0;k<dim;k++){
			  big_bord[i][j][k]=0.0;
			  small_bord[i][j][k]=0.0;
		  }
	  }
  }
  int maxpoints[dim], ntotal = rempoints + nforced;  
  // biggest[i][j]: biggest product of coords 0--i for hitting j points
  // smallest[i][j]: smallest product of coords 0--i for hitting j+1 points
  // maxpoints[i]: number of points you get in total from coords 0--i
  double vol1=1.0, vol2=1.0;
  for (i=0; i<dim; i++) {
    vol1 *= lowerleft[i];
    vol2 *= upperright[i];
  }    
#ifdef SPAM
  fprintf(stderr, "Parameters: npoints %d, dim %d, rempoints %d, nforced %d\n",
	  npoints, dim, rempoints, nforced);
  fprintf(stderr, "Lower: (%g", lowerleft[0]);
  for (i=1; i<dim; i++)
    fprintf(stderr, ", %g", lowerleft[i]);
  fprintf(stderr, ")\nUpper: (%g", upperright[0]);
  for (i=1; i<dim; i++)
    fprintf(stderr, ", %g", upperright[i]);
  fprintf(stderr, ")\nUncategorized ('forced') points are:\n");
  for (i=0; i<nforced; i++) {
    fprintf(stderr, "(%g", forced[i][0]);
    for (j=1; j<dim; j++)
      fprintf(stderr, ", %g", forced[i][j]);
    fprintf(stderr, ")\n");
  }
#endif
  //fprintf(stderr, "Get in cell\n");
  maxdiscr = vol2 - (double)rempoints/npoints;
  discr = (double)(rempoints+nforced)/npoints - vol1;
  if (discr > maxdiscr){
    maxdiscr = discr;
	for (i=0;i<dim;i++)
		curr_bord[i]=upperright[i]; 
	temp_fill=true;
	
  }
  if (maxdiscr < globallower){
	  for (i=0;i<dim;i++){
	  for (j=0;j<(nforced+1);j++){
		  free(small_bord[i][j]);
		  free(big_bord[i][j]);
	  }
	  free(big_bord[i]);
	  free(small_bord[i]);
  }
  free(big_bord);
  free(small_bord);
    return maxdiscr;
  }
  // quicker code for use in some easy cells
  // otherwise, get to work...
  for (i=0; i<dim; i++) {
    indexes[i]=0;
    for (j=0; j<=nforced; j++) {
      smallest[i][j]=1.0;
      biggest[i][j]=0.0;
    }
  }
  for (i=0; i<nforced; i++) {
    status=0;
    for (j=0; j<dim; j++) {
      // order is chosen to handle final box
      if (forced[i][j] <= lowerleft[j])
	continue; 
      else if (forced[i][j] >= upperright[j])
	break;
      else { // strictly internal
	if (status) {
	  fprintf(stderr, "PROBLEM: Point occurs as double-internal\n");
	  fflush(stderr);
	  abort();
	}
	status = 1;
	dimension=j;
      }
    }
    if (j==dim) { // else: hit "break", skip
      if (status) {
	coordlist[dimension][indexes[dimension]]=forced[i][dimension];
	indexes[dimension] += 1;
      }
      else { // completely internal
	rempoints++;
      }
    }
  }
  
  for (i=0; i<dim; i++)
    qsort(&(coordlist[i][0]), indexes[i], sizeof(double), cmpdbl);
  maxpoints[0]=indexes[0];
  for (i=1; i<dim; i++)
    maxpoints[i]=maxpoints[i-1]+indexes[i];
#ifdef SPAM
  fprintf(stderr, "Categorization: %d lower-left, %d internal.\n", rempoints, maxpoints[dim-1]);
  for (i=0; i<dim; i++) {
    if (!indexes[i]) {
      fprintf(stderr, "Direction %d: Nothing.\n", i);
      continue;
    }
    fprintf(stderr, "Direction %d: %g", i, coordlist[i][0]);
    for (j=1; j<indexes[i]; j++)
      fprintf(stderr, ", %g", coordlist[i][j]);
    fprintf(stderr, "\n");
  }
#endif
  
  // coord 0 first, since there is no recursion for that:
  smallest[0][0]=lowerleft[0];
  small_bord[0][0][0]=lowerleft[0]; //Changed here
  for (j=0; j<indexes[0]; j++) {
    smallest[0][j+1]=coordlist[0][j];
	biggest[0][j]=coordlist[0][j];
	small_bord[0][j+1][0]=coordlist[0][j]; // Changed here
	big_bord[0][j][0]=coordlist[0][j];
  }
  biggest[0][indexes[0]]=upperright[0];
  big_bord[0][indexes[0]][0]=upperright[0];
  
  //INIT CORRECT
  
  
  
  
#ifdef SPAM
  fprintf(stderr, "Direction 0 only, biggest: ");
  for (j=0; j<=maxpoints[0]; j++)
    fprintf(stderr, "%g ", biggest[0][j]);
  fprintf(stderr, "\nDirections 0 only, smallest: ");
  for (j=0; j<=maxpoints[0]; j++)
    fprintf(stderr, "%g ", smallest[0][j]);
  fprintf(stderr, "\n");
#endif
    
  for (i=1; i<dim; i++) {
    // first the special loop for smallest: "nothing contributed by us"
    for (j=0; j<=maxpoints[i-1]; j++){
      smallest[i][j]=smallest[i-1][j]*lowerleft[i];
	  for (h=0;h<i;h++)
		  small_bord[i][j][h]=small_bord[i-1][j][h];
	  small_bord[i][j][i]=lowerleft[i];
	}
    // main loop:
    for (j=0; j<indexes[i]; j++) {
      vol1 = coordlist[i][j];
      for (k=0; k<=maxpoints[i-1]; k++) {
	// for biggest: vol1 is coordinate that adds j new points
	vol2=biggest[i-1][k]*vol1;
	if (vol2 > biggest[i][j+k]){
	  biggest[i][j+k]=vol2;
  
    for (h=0;h<i;h++)
		  big_bord[i][j+k][h]=big_bord[i-1][k][h]; // Copy the values we had obtained for this pre-set before. yes, memcpy would be better.
	big_bord[i][j+k][i]=coordlist[i][j]; // Add the new dimension that wasn't known beforehand.
	}
	// for smallest: vol1 is coordinate that adds j+1 new points
	vol2=smallest[i-1][k]*vol1;
	if (vol2 < smallest[i][j+k+1]){
		for (h=0;h<i;h++)
		  small_bord[i][j+k+1][h]=small_bord[i-1][k][h]; // Changed here
		small_bord[i][j+k+1][i]=coordlist[i][j]; // Changed here
	  smallest[i][j+k+1]=vol2;
		}
      }
    }
    // last, special loop for biggest: "all of us"
    for (j=0; j<=maxpoints[i-1]; j++) {
      vol1=biggest[i-1][j]*upperright[i];
      if (vol1 > biggest[i][j+indexes[i]]){
	biggest[i][j+indexes[i]]=vol1;
	for (h=0;h<i;h++)
		  big_bord[i][j+indexes[i]][h]=big_bord[i-1][j][h];
	  big_bord[i][j+indexes[i]][i]=upperright[i];
	  }
    }
#ifdef SPAM
    fprintf(stderr, "Directions 0--%d, biggest: ", i);
    for (j=0; j<=maxpoints[i]; j++)
      fprintf(stderr, "%g ", biggest[i][j]);
    fprintf(stderr, "\nDirections 0--%d, smallest: ", i);
    for (j=0; j<=maxpoints[i]; j++)
      fprintf(stderr, "%g ", smallest[i][j]);
    fprintf(stderr, "\n");
#endif
  }
  
  // now, use these to find lower, upper limits
  // mode: always contain "rempoints", additionally 
  //         pick from smallest[dim-1], biggest[dim-1]
  maxdiscr=0;
  //fprintf(stderr, "DynProg time\n");
  for (i=0; i<=maxpoints[dim-1]; i++) { // i counts internal points
    // small box
	//fprintf(stderr, "%d\n",i);
    discr = (double)(rempoints+i)/npoints - smallest[dim-1][i];
    if (discr > maxdiscr){
		
      maxdiscr=discr;
	  for (j=0;j<dim;j++)
		  curr_bord[j]=small_bord[dim-1][i][j]; // Now storing solution if it's a small box
	  temp_fill=true;
	  
	  
	}
    // big box
    discr = biggest[dim-1][i] - (double)(rempoints+i)/npoints;
	//fprintf(stderr, "Hi2\n");
    if (discr > maxdiscr){
      maxdiscr = discr;
	  //fprintf(stderr, "Hi3\n");
	  for (j=0;j<dim;j++){
		  curr_bord[j]=big_bord[dim-1][i][j]; // Now storing solution if it's a big box
	  }
	  temp_fill=false;
	  
	}
  }
  //fprintf(stderr, "Get out cell2\n");
  if (maxdiscr > globallower) {
#ifdef WORK_OUTPUT
    fprintf(stderr, "Worse bound: %g\n", maxdiscr);
#endif
    globallower=maxdiscr;
  }
#ifdef SPAM
  else
    //fprintf(stderr, "Conclusion: %g\n", maxdiscr);
#endif
	//fprintf(stderr, "left a cell\n");
	
  /*if (debugg<3){
	  for (i=0;i<dim;i++){
		  for (j=0;j<nforced+1;j++){
			  for (h=0;h<dim;h++)
				  fprintf(stderr,"%lf, ",big_bord[i][j][h]);
			  fprintf(stderr," New j \n");
		  }
		  fprintf(stderr," New i \n");
	  }
	  fprintf(stderr,"Small now \n");
	  for (i=0;i<dim;i++){
		  for (j=0;j<nforced+1;j++){
			  for (h=0;h<dim;h++)
				  fprintf(stderr,"%lf, ",small_bord[i][j][h]);
			  fprintf(stderr," New j \n");
		  }
		  fprintf(stderr," New i \n");
	  }
	  fprintf(stderr,"\n");
	  debugg+=1;
  }	  */
  for (i=0;i<dim;i++){
	  for (j=0;j<(nforced+1);j++){
		  free(small_bord[i][j]);
		  free(big_bord[i][j]);
	  }
	  free(big_bord[i]);
	  free(small_bord[i]);
  }
  free(big_bord);
  free(small_bord);
	
  return maxdiscr;
}

// "forced" points are points that are strictly between the boundaries in
// one of the cdim earlier dimensions; pointset[0--rempoints-1] are points that
// so far are at most at the lower-left corner in every dimension.
// this includes a final run with lower-left=1.
// being ON a border changes nothing:
//   ON lower-left counts as in (including when lowerleft=1)
//   ON upper-right counts as out (except if previous).
double oydiscr_int(double **pointset, int npoints, int dim, int rempoints,
		   double **forced, int nforced, int cdim, 
		   double *lowerleft, double *upperright)
{
  double coord, forcedcoord, lowedge=0.0, highedge;
  //int comparedim;
  int newcount=0, forcedidx, i, j, previdx=0, newforcedidx, resort=0, curridx;
  int newrempoints, wasforced, wasfinal=0;
  double maxdiscr=0.0, discr;
  double **newforced = malloc((nforced+rempoints)*sizeof(double *));
  // internal vars: previdx points at first element excluded from last pass
  //                    (on or above border coordinate)
  //                forcedidx: next unused forced element (a.k.a. counter)
  //                curridx: current pointset point ("i" as loop var)
  //  coords:
  //           coord is value of current pointset-point, 
  //           forcedcoord is value of next-up forced boundary,
  //           lowedge is value of last boundary we used
  // newcount counts number of pointset-points since last coord
  if (cdim==dim) {
	  //fprintf(stderr, "Reached a cell\n");
    free(newforced);
    discr= oydiscr_cell(npoints, dim, rempoints, 
			forced, nforced,
			lowerleft, upperright);
			double check;
	if (discr>glob_bound){
		for (j=0;j<dim;j++)
			best_bord[j]=curr_bord[j];
	    if (temp_fill)
			curr_fill=true;
	    else
			curr_fill=false;
		glob_bound=discr;
	}
	return discr;
	
  }
  
  comparedim=cdim;
  qsort(pointset, rempoints, sizeof(double *), cmpkeyk);
  qsort(forced, nforced, sizeof(double *), cmpkeyk);
  i=0; forcedidx=0;
  while ((i<rempoints) || (forcedidx < nforced)) {
    if (i<rempoints)
      coord=pointset[i][cdim];
    else
      coord=1.0;
    if (forcedidx < nforced)
      forcedcoord=forced[forcedidx][cdim];
    else
      forcedcoord=1.0;
    if ((forcedcoord > coord) && (newcount <= sqrt(npoints))) {
      newcount++;
      i++;
      if ((i<rempoints) || (forcedidx < nforced))
	continue;
      else { // Add one trailing cell
	lowerleft[cdim]=lowedge;
	highedge=upperright[cdim]=1.0;
	wasforced=0;
	wasfinal=1;
      }
    } // below: create new cell
    if (!wasfinal) {
      if (forcedcoord <= coord) {
	lowerleft[cdim]=lowedge;
	highedge=upperright[cdim]=forcedcoord;
	wasforced=1;
      }
      else { // must be count-based border
	lowerleft[cdim]=lowedge;
	highedge=upperright[cdim]=coord;
	wasforced=0;
      }
    } // end "if (!wasfinal)"
    curridx=i; // for better mnemonics
#ifdef WORK_OUTPUT
    if (!cdim)
      fprintf(stderr, "Coord %g\n", highedge);
#endif
    // creating a new cell (or subslab):
    // 1. surviving forced copied
    for (j=0; (j<nforced) && (forced[j][cdim] < highedge); j++)
      newforced[j]=forced[j];
    newforcedidx=j;
    // 2. new (strictly) internal points appended as forced
    j=previdx;
    while ((j<rempoints) && (pointset[j][cdim] <= lowedge))
      j++;
    newrempoints=j;
    for (; (j<rempoints) && (pointset[j][cdim] < highedge); j++)
      newforced[newforcedidx++] = pointset[j];
    if (j>(curridx+1))
      resort=1; 
    // 3. make call with properly adjusted boundaries, update variables
    discr = oydiscr_int(pointset, npoints, dim, newrempoints,
			newforced, newforcedidx, cdim+1,
			lowerleft, upperright);
    if (discr > maxdiscr) {
      maxdiscr = discr;
	  
	  /*for (j=0;j<dim;j++)
		  best_bord[j]=curr_bord[j];
	  if (curr_fill)
		fill=true;
	  else
		fill=false;*/
	  
	}
    if (resort) {
      comparedim=cdim;
      qsort(pointset, rempoints, sizeof(double *), cmpkeyk); // HERE!!!!!
      resort=0;
    }
    while ((forcedidx < nforced) && (forced[forcedidx][cdim]==highedge))
      forcedidx++;
    while ((i < rempoints) && (pointset[i][cdim]<=highedge))
      i++;
    lowedge=highedge;
    previdx=i;
    newcount=0;
  }
  // one final call to capture the border cases (for boxes containing a point with coord 1.0)
  // 1. new forced == old forced (copy to avoid interfering with previous stages)
  for (j=0; j<nforced; j++)
    newforced[j]=forced[j];
  // 2. per above, we have no new internal/forced points
  // 3. make the call
  lowerleft[cdim]=lowedge;
  upperright[cdim]=1.0;
  discr = oydiscr_int(pointset, npoints, dim, rempoints,
		      newforced, nforced, cdim+1,
		      lowerleft, upperright);
  if (discr>maxdiscr){
    maxdiscr=discr;
	/*for (j=0;j<dim;j++)
		best_bord[j]=curr_bord[j];
	if (curr_fill)
		fill=true;
	else
	    fill=false;*/
	
  }
	
  free(newforced);
  return maxdiscr;


}
    
double oydiscr(double **pointset, int dim, int npoints, double *lower)
{
  double lowerleft[dim], upperright[dim];
  double **pre_force = malloc(2*dim*sizeof(double *));
  double discr, *border;
  double maxcoord;
  int maxpos;
  int is_border[npoints];
  double **clone_set = malloc(npoints*sizeof(double *));
  //double *best_bord=malloc(dim*sizeof(double)); // Moved to param
  // No need to initialize them, values get replaced depending only on discr.
  int i,j,k;
  //fprintf(stderr,"Going to int");
  for (i=0; i<dim; i++) {
    border = malloc(dim*sizeof(double));
    for (j=0; j<dim; j++) {
      if (i==j)
	border[j]=1.0;
      else
	border[j]=0.0;
    }
    pre_force[i]=border;
  }
  for (i=0; i<npoints; i++)
    is_border[i]=0;
  for (i=0; i<dim; i++) {
    maxcoord=-1.0;
    maxpos=-1;
    for (j=0; j<npoints; j++) 
      if (pointset[j][i] > maxcoord) {
	maxcoord = pointset[j][i];
	maxpos=j;
      }
    is_border[maxpos]=1;
  }
  j=dim; k=0;
  for (i=0; i<npoints; i++)
    if (is_border[i])
      pre_force[j++]=pointset[i];
    else
      clone_set[k++]=pointset[i];
//  discr = oydiscr_int(pointset, npoints, dim, npoints,
//		      pre_force, dim, 0, 
//		      lowerleft, upperright);
//  discr = oydiscr_int(clone_set, npoints, dim, k,
//		      pre_force, j, 0,
//		      lowerleft, upperright);
//  discr = oydiscr_int(clone_set, npoints, dim, k,
//		      pre_force[dim], j-dim, 0,
//		      lowerleft, upperright);
  // final version: NOTHING pre-determined.
  //fprintf(stderr, "First part\n");
  discr = oydiscr_int(pointset, npoints, dim, npoints, 
		      pre_force, 0, 0,
		      lowerleft, upperright); // Careful, this is called npoints like in original code but is equal to kpoints
  for (i=0; i<dim; i++)
    free(pre_force[i]);
  free(pre_force);
  free(clone_set);
  *lower = globallower;
  //fprintf(stderr,"%lf\n",discr);
  return discr;
}


//////////////////////////////////////////////////////////////////////


int find_point(double **copysorted,double temp_coord, int cho, int npoints){
	int a,b,mid;
	a=0;
	b=npoints-1;
	bool found;
	found=false;
	mid=(a+b)/2;
	while (b-a>1 && !found){
		if (copysorted[cho][mid]== temp_coord){
			return(mid);
		}
		else if (copysorted[cho][mid]< temp_coord){
			a=mid;
			mid=(a+b)/2;
		}
		else {
			b=mid;
			mid=(a+b)/2;
		}
			 
	}
	if (b-a==1){ // This only works because we KNOW the box should be given by some coord. With double precision there might be some mistake in the volume calc-> inexact coord. We take the closest one.
		if (copysorted[cho][b]-temp_coord> temp_coord-copysorted[cho][a])
			return b;
		else
			return a;
	}
	return -1;
}


// ADD THIS AFTER EACH DISCRE CALC!!
void replace(double **pointset, double **Orig_pointset, int kpoints, int npoints){
	int i,j;
	for(i=0;i<npoints;i++)
		is_in[i]=-1;
	for (i=0;i<kpoints;i++){
		for (j=0; j<npoints;j++){
			if (fabs(pointset[i][0]-Orig_pointset[j][0])<epsilon/2){
				is_in[j]=i;
			}
		}
	}
	return;
	
}



double shift(int npoints, int kpoints, int dim, double **Orig_pointset)
{
  
  // SHuffle our point array, shuffle directly in Orig_pointset. Do this in main?
  
  
  int i, j, h, a, b, c, d, f, g, search_count, temp_bordpoint, chosen_dim, nb_natural, nb_brute, nb_runs, tempo, curr_dim, actu_dim, index;
  double upper,lower;
  upper=1.0;
  int nb_calc=0;
  
	
// Sorting the points in each dim
  double **copysorted=malloc(dim*sizeof(int*));
  double **subset;
  double **temp_subset;
  int **orderings=malloc(dim*sizeof(int*));// The ordering[i][j]=h means that point j is h-th in the ordering in dimension i.
  int **revorderings=malloc(dim*sizeof(int*)); // revordering[i][j]=h means that the j-th point in dimension i is point h.
  for (i=0;i<dim;i++){
	  copysorted[i]=malloc(npoints*sizeof(double));
	  orderings[i]=malloc(npoints*sizeof(int));
	  revorderings[i]=malloc(npoints*sizeof(int));
	  for (j=0;j<npoints;j++)
		  copysorted[i][j]=Orig_pointset[j][i]; // Warning dimensions switched
	  qsort(&(copysorted[i][0]), npoints, sizeof(double), cmpdbl);
	  for (j=0;j<npoints;j++){// Need points in general position for this
		  for (h=0;h<npoints;h++){
			  if (copysorted[i][j]==Orig_pointset[h][i]){ // Same as above
				  orderings[i][h]=j;
				  revorderings[i][j]=h;
				  break; // We supposed general position
			  }
		  }	  
	  } 
  }
  is_in =malloc(npoints*sizeof(int));
  for (i=0;i<npoints;i++){
	  if (i<kpoints)
		is_in[i]=i;
	  else
		is_in[i]=-1;
	}
	subset=malloc(kpoints*sizeof(double*)); // INTRODUCE kpoints
	temp_subset=malloc(kpoints*sizeof(double*));
	for (i=0;i<kpoints;i++){// Our current point set and create a future tep_subset for the possible changes.
		subset[i]=malloc(dim*sizeof(double));
		temp_subset[i]=malloc(dim*sizeof(double));
		memcpy(subset[i],Orig_pointset[i], dim*sizeof(double));
		memcpy(temp_subset[i],Orig_pointset[i], dim*sizeof(double));
	}
	fprintf(stderr, "Sorted points\n");
	upper = oydiscr(temp_subset, dim, kpoints,&lower); // We know this is "optimal" at least for the moment
	glob_bound=0;
	
	lower=0.0;
	nb_calc+=1;
	globallower=0.0;
	//replace(subset,Orig_pointset,kpoints,npoints); 
	double curr_disc;
	bool chosen,boom,problem;
	nb_natural=0;nb_brute=0;
	double *top_bord;
	top_bord=malloc(dim*sizeof(double));
	//memcpy(top_bord,best_bord,dim*sizeof(double));
	for (i=0;i<dim;i++)
		top_bord[i]=best_bord[i];
	bool insidei, insidej;
	// Following initialisations shouldn't be necessary?
	temp_bordpoint=-1;
	curr_disc=1.0;
	nb_runs=1000; // Tweak it here directly (useless for the moment)
	problem=false;
	int *list_points;
	list_points=malloc(dim*sizeof(int));
	//
	for (b=0; b<nb_runs;b++){
		chosen_dim=rand() %dim;
		chosen=false;
		for (i=0;i<dim;i++){
			temp_bordpoint=find_point(copysorted,top_bord[i],i,npoints);
			c=revorderings[i][temp_bordpoint];
			if (is_in[c]< -0.5){
				problem=false;
				index=temp_bordpoint;
				while(index >=0 && is_in[revorderings[i][index]]< -0.5 )
					index--;
				if (index==-1){
					problem=true;
					break;
				}
				c=revorderings[i][index];
			}
			list_points[i]=c;
		}
		search_count=0;
		actu_dim;
		while (!chosen && search_count<npoints && !problem){ // bound on search_count could be improved
			curr_dim=0;
			if (fill){
				g=0;
				while (!chosen && curr_dim<dim){
					actu_dim=(curr_dim+chosen_dim)%dim;
					c=orderings[actu_dim][list_points[actu_dim]]; // The position of the point we want to replace. Could maybe pre-define a table to avoid recomputing this every time
					if (c+search_count>=npoints){
						g+=1;
						curr_dim+=1;
						if (g==dim) // We're stuck, no need to go further
							break;
						else
							continue;
					}
					d=revorderings[actu_dim][c+search_count]; // Candidate for replacement
					if (is_in[d]< -0.5){// The point was not already in the set
						c=list_points[actu_dim]; // Before we had the position and not the point number
						for (i=0;i<kpoints;i++)
							memcpy(temp_subset[i],subset[i],dim*sizeof(double));
						tempo=is_in[c];
						memcpy(temp_subset[tempo],Orig_pointset[d],dim*sizeof(double));
						curr_disc = oydiscr(temp_subset, dim, kpoints, &lower);
						glob_bound=0;
						lower=0.0;
						globallower=0.0;
						nb_calc+=1;
						if (curr_disc<upper) {// Our replacement is good
							chosen=true;
							nb_natural+=1;
							memcpy(subset[tempo],Orig_pointset[d],dim*sizeof(double));
							is_in[d]=tempo;
							is_in[c]=-1;
							fill=curr_fill;
							memcpy(top_bord,best_bord,dim*sizeof(double));
							upper=curr_disc;
							fprintf(stderr,"New:%lf",upper);
						}
						//BUG WAS HERE, WITH AN ELSE MODIFYING OUR SUBSET FOR NO REASON
						}
				curr_dim+=1;	
					}
			}
			
			else{ // Now underfilled box
				g=0;
				while (!chosen && curr_dim<dim){
					actu_dim=(curr_dim+chosen_dim)%dim;
					c=orderings[actu_dim][list_points[actu_dim]]; // The point we want to replace
					if (c-search_count<0){
						g+=1;
						curr_dim+=1;
						if (g==dim) // We're stuck
							break;
						else
							continue;
					}
					d=revorderings[actu_dim][c-search_count];
					if (is_in[d]< -0.5){// The point was not already in the set
						c=list_points[actu_dim]; // Before we had the position and not the point number
						for (i=0;i<kpoints;i++)
							memcpy(temp_subset[i],subset[i],dim*sizeof(double));
						tempo=is_in[c];							
						memcpy(temp_subset[tempo],Orig_pointset[d],dim*sizeof(double));
						curr_disc = oydiscr(temp_subset, dim, kpoints, &lower);
						glob_bound=0;
						lower=0.0;
						globallower=0.0;
						nb_calc+=1;
						if (curr_disc<upper) {// Our replacement is good
							chosen=true;
							nb_natural+=1;
							memcpy(subset[tempo],Orig_pointset[d],dim*sizeof(double));
							is_in[d]=tempo;
							is_in[c]=-1;
							
							fill=curr_fill;
							memcpy(top_bord,best_bord,dim*sizeof(double));
							
							upper=curr_disc;
							fprintf(stderr,"New2:%lf",upper);
						}
					}
						curr_dim+=1;
				}
			}
		search_count+=1; // We've gone thourgh a full dim rotation or found a point	
		}
		
	if (!chosen)
		break;
	}
	fprintf(stderr,"Nb calcu:%d, Final discre:%lf\n",nb_calc,upper);
	for (i=0;i<kpoints;i++)
		memcpy(optiset[i],subset[i], dim*sizeof(double));
		
	for (i=0;i<dim;i++){
		free(copysorted[i]);
		free(orderings[i]);
		free(revorderings[i]);
	}
	for (i=0;i<kpoints;i++){
		free(subset[i]);
		free(temp_subset[i]);
	}
	free(copysorted);
	free(subset);
	free(temp_subset);
	free(orderings);
	free(revorderings);
	free(is_in);
	free(list_points);
	
	free(top_bord);
	
  fprintf(stderr,"Natural: %d, Brute: %d, Discr: %lf",nb_natural,nb_brute,upper);
  return upper;

  
}

int main(int argc, char **argv)
{
  int dim, npoints,i,j,h,kpoints;
  int chosen_dim;
  FILE *pointfile;
  double upper,lower;
  int nb_tries;
  bool fill; //Tracks if the worst box was under or over filled
  double **Orig_pointset;
  FILE *random;
  unsigned int seed;
  srand(1);
  pointfile = fopen(argv[1], "r");
  /*int test;
  test=fscanf(pointfile,"%d %d %d", &dim, &npoints, &kpoints);
  if (test!=3)
	  exit(EXIT_FAILURE);
  fprintf(stderr, "Reading dim %d npoints %d kpoints %d\n", dim, npoints,kpoints);*/
  
  dim=atoi(argv[2]);
  npoints=atoi(argv[3]);
  kpoints=atoi(argv[4]);
  
  fprintf(stderr, "\n \n Reading dim %d npoints %d kpoints %d ", dim, npoints,kpoints);
  
  Orig_pointset = malloc(npoints*sizeof(double*));
  for (i=0; i<npoints; i++) {
    Orig_pointset[i] = malloc(dim*sizeof(double));
    for (j=0; j<dim; j++) {
      // newline counts as whitespace
      if (!fscanf(pointfile, "%lg ", &(Orig_pointset[i][j]))) {
	fprintf(stderr, "File does not contain enough data points!\n");
	exit(EXIT_FAILURE);
      }
    }
  }
  double super_best=1.0;
  int tries;
  char *tries_env = getenv("SHIFT_TRIES");
  if (tries_env != NULL) {
	tries = atoi(tries_env);
	if (tries <= 0) {
	  fprintf(stderr, "Invalid value for TRIES: %s\n", tries_env);
	  exit(EXIT_FAILURE);
	}
  }
  else {
	tries = 50;
  }
  for (nb_tries=0;nb_tries<tries;nb_tries++){
  int rando1;
  int rando2;
  double *swap;
  swap=malloc(dim*sizeof(double));
  //SHUFFLING TIME: WORKS WITHOUT
  for (i=0;i<10*npoints;i++){
	  start=clock();
	  rando1= rand() % npoints;
	  rando2=rand() % npoints;
	  for (j=0;j<dim;j++){
		  swap[j]=Orig_pointset[rando1][j];
		  Orig_pointset[rando1][j]=Orig_pointset[rando2][j];
		  Orig_pointset[rando2][j]=swap[j];
  }
  }
  
  free(swap);
  epsilon=1;
  for (i=0;i<npoints;i++){
	  for (j=i+1;j<npoints;j++){
		  if (fabs(Orig_pointset[i][0]-Orig_pointset[j][0])<epsilon)
			  epsilon=fabs(Orig_pointset[i][0]-Orig_pointset[j][0]);
	  }
  }
  best_bord=malloc(dim*sizeof(double));
  curr_bord=malloc(dim*sizeof(double));
  
  
	for (int i=0;i<dim;i++){ // Shouldn't be necessary as oydiscr necessarily modifies this at some point?
		best_bord[i]=0;
		curr_bord[i]=0;
	}
  optiset=malloc(kpoints*sizeof(double*));  
  fill=false; // Should be useless
  for (i=0;i<kpoints;i++){
	  optiset[i]=malloc(dim*sizeof(double));
	  memcpy(optiset[i],Orig_pointset[i], dim*sizeof(double));
  }
   
  upper=shift(npoints,kpoints,dim,Orig_pointset);
  end = clock();
  cput = ((double) (end - start)) / CLOCKS_PER_SEC;
  if (super_best>upper){
	  super_best=upper;
  FILE *fp; // Move our opti point set to a file
  fp=fopen(argv[5],"w");
  fprintf(fp, "n=%d,k=%d,dim=%d, discrepancy=%lf, runtime=%lf\n",npoints,kpoints,dim,upper,cput);
  for (i=0; i<kpoints;i++){
	  for (j=0;j<dim;j++){
		  fprintf(fp,"%.18e ",optiset[i][j]);
	  }
	  fprintf(fp,"\n");
  }
  fclose(fp);
  }
  for(i=0;i<kpoints;i++)
	  free(optiset[i]);
  free(optiset);
  }
  fclose(pointfile);
  for (i=0;i<npoints;i++)
	  free(Orig_pointset[i]);
  free(Orig_pointset);
  free(curr_bord);
  free(best_bord);
  fprintf(stderr,"We're done!");
  return 0;

}
