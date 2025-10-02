#include <omp.h>
#include <quadmath.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

double eps = 1.0e-5;   // values below this is are treated as zero

int printonce = 1;

double psi12 , psi13, psi23;
double d120, d130, d230;

double minsigmaabssdot = 1.0e100;
double minsigmat       = 0.0;

double integrationinterval;

double PI = acos(-1.0);

// massies at triangle vertex points
double m1 = 2.1;
double m2 = 2.1;
double m3 = 4.7;
double mass[4];

double inertia;

double diffpsi; // \phi_{13} - \phi_{23} as in Fig. 1

double derd12, derd13, derd23;

int main(int argc, char *argv[])
    {
    double qromo(double (*func)(double,double), double a, double b,
        double (*choose)(double(*)(double,double), double, double, int, double, int),double, int);
    double dthetadt(double t, double h);
    double dthetadtsvalues(double t, double h, double scoord[][4], double sder[][4]);
    double midpnt(double (*func)(double,double), double a, double b, int n, double delta, int id);
    int func(double t, double shapecoord[][4]);

    double globalstart         = 0.0;
    double periodend           = 0.00021;
    double globalend           = 0.00021*200; // 200 common periods
    integrationinterval        = 0.000070/10.0; // T_{12}
    mass[1] = m1;
    mass[2] = m2;
    mass[3] = m3;

    printf("   Integrating %d T_{12} periods\n",(int)((globalend-globalstart)/(periodend-globalstart)+0.5));

    int numofintegrationintervals = (int)((globalend - globalstart)/integrationinterval+0.5);

    double h = integrationinterval/10.0;
    printf("   t runs %.10e -> %.10e\n", globalstart, globalstart + integrationinterval*((double)numofintegrationintervals));
    printf("   integrationinterval = %g, # of intervals = %d\n",integrationinterval, numofintegrationintervals);
    printf("   derivative approximated using interval [%g,+%g]\n",-h,h);

    double totalintegral;

    FILE *outfil = fopen("psi.out","w");
    if ( outfil == NULL ){printf("Cannot open output file psi.out\n");return(-1);}
    fprintf(outfil,"##   this is a plot of \\theta(200T) vs. \\psi = \\phi_{13} - \\phi_{23}\n");
    fprintf(outfil,"##   \\psi runs from 0 to 2*\\pi\n");
    fprintf(outfil,"##   \\phi_{13} is fixed at \\pi/4.0 and \\phi_{23} is adjusted for psi\n");

    for ( int i = 0 ; i <= 1000; ++i ) 
        {
        diffpsi = 0.0 + i*2.0*PI/1000.0;
        totalintegral = 0.0;
        for ( int i = 1; i <= numofintegrationintervals; ++i )
            {
            double start = globalstart + (i-1)*integrationinterval;
            double end   = start + integrationinterval;
            if ( i == numofintegrationintervals ) end = globalend;
            double result = qromo(dthetadt, start, end, midpnt, h, 0 );
            totalintegral += result;
            }
	 // print every 30th value on screen for observation of progress
         if ( i%30 == 0 ) printf("   %lf\t\t%.10g\n",diffpsi, (double)totalintegral);
         fprintf(outfil,"%lf\t%.10g\n",diffpsi, (double)totalintegral);
         }
    fclose(outfil);
     
    return(0); 
    }

int func(double t, double shapecoord[][4])
            {
	    double twopi = 2.0*acos(-1.0);
	    double a12, a13, a23;
            double T12, T13, T23;
	    double d12, d13, d23;
	    double d12max, d13max, d23max;
	    double d12min, d13min, d23min;
	    double t12_1, t12_2, t13_1, t13_2, t23_1, t23_2;
	    double w12_1, w12_2, w13_1, w13_2, w23_1, w23_2;
	    double deltat;

            d120  = 1.1;
	    d130  = 1.0;
	    d230  = 1.0;
	    a12   = 0.2;
	    a13   = 0.15;
	    a23   = 0.15;
            T12   = 0.000070;
            T13   = 0.000210;
            T23   = 0.000210;
	    psi12 = 0.0;
	    psi13 = 3.141592654/4.0; // pi/4.0
	    psi13 = diffpsi/2.0;
	    psi23 = -psi13;    

	    d12 = d120 + a12*cos(twopi*t/T12+psi12);   //distances as functions of time
	    d13 = d130 + a13*cos(twopi*t/T13+psi13);   //distances as functions of time
	    d23 = d230 + a23*cos(twopi*t/T23+psi23);   //distances as functions of time

	    derd12 =  -twopi/T12*a12*sin(twopi*t/T12+psi12);   //derivate of bond length as a function of time
	    derd13 =  -twopi/T13*a13*sin(twopi*t/T13+psi13);   //derivate of bond length as a function of time
	    derd23 =  -twopi/T23*a23*sin(twopi*t/T23+psi23);   //derivate of bond length as a function of time


	    if ( d12 < 0.0 ) {printf("   Negative distance d12 %lf at time %e\n",d12,t);exit(1);}
	    if ( d13 < 0.0 ) {printf("   Negative distance d13 %lf at time %e\n",d13,t);exit(1);}
	    if ( d23 < 0.0 ) {printf("   Negative distance d23 %lf at time %e\n",d23,t);exit(1);}

	    if ( d12 + d23 < d13 ) 
	       {
	       printf(" triangle inequality not true at t = %e : d12 %lf + d23 %lf should be > d13 %lf\n", t,d12,d23,d13);
	       exit(1);
	       }
	    if ( d23 + d13 < d12 ) 
	       {
	       printf(" triangle inequality not true at t = %e  : d23 %lf + d13 %lf should be >  d12 %lf\n", t,d23,d13,d12);
	       exit(1);
	       }
	    if ( d12 + d13 < d23 ) 
	       {
	       printf(" triangle inequality not true at t = %e : d12 %lf + d13 %lf  should be >  d23 %lf\n", t,d12,d13,d23);
    	       exit(1);
	       }
               

	    if ( printonce == 1 )
	       {
	       printonce = 0;
	       printf("   Model parameters:\n");
	       printf("\t d120 = %lf\t d130 = %lf\t d230 = %lf\n",d120,d130,d230);
	       printf("\t  a12 = %lf\t  a13 = %lf\t  a23 = %lf\n",a12,a13,a23);
	       printf("\t  T12 = %lf\t  T13 = %lf\t  T23 = %lf\n",T12,T13,T23);
	       printf("\tpsi12 = %lf\tpsi13 = %lf\tpsi23 = %lf\n",psi12,psi13,psi23);
	       printf("\t   m1 = %lf\t   m2 = %lf\t   m3 = %lf\n",m1,m2,m3);
               printf("  \\phi_{13}-\\phi_{23}\t\\theta\n");
	       }

	    // formulating M_ij(t) = (d_1j^2(t) + d_i1^2(t) - d_ij^2(t) )/2.0
	    double M22 = d12*d12;
	    double M23 = (d12*d12+d13*d13-d23*d23)/2.0;
	    double M33 = d13*d13;

	    // solving for eigenvalues lambda1,2 of M
	    double p = -(M22+M33);
	    double q = M22*M33-M23*M23;
	    double lambda1 = -p/2.0+sqrt(p*p/4.0-q);
	    double lambda2 = -p/2.0-sqrt(p*p/4.0-q);
	    if ( lambda1 < 0.0 || lambda2 < 0.0 ) exit(1);;

	    double x1,y1,x2,y2;

	    // finding normalized eigenvectors of M, (x1,y1) and (x2,y2)
            if ( fabs(M23) < eps ) 
	       {
	       if ( fabs(M22-lambda1) < eps ) { x1 = 1.0; y1 = 0.0; }
	       else { x1 = 0.0; y1 = 1.0; }
	       }
	    else 
	        {
		y1 = 1.0; 
		x1 = -M23/(M22-lambda1);
	        double len = sqrt(x1*x1+y1*y1);
	        x1 = x1/len;
	        y1 = y1/len;
	        }
            
            if ( fabs(M23) < eps ) 
	       {
	       if ( fabs(M22-lambda2) > eps ) { x2 = 0.0; y2 = 1.0; }
	       else  {x2 = 1.0; y2 = 0.0; }
	       }
	    else 
	        {
		y2 = 1.0; 
		x2 = -M23/(M22-lambda2);
	        double len  = sqrt(x2*x2+y2*y2);
	        x2 = x2/len;
	        y2 = y2/len;
	        }
	    

	    //checking that U^T * U = I
	    // U = eigenvectors as columns
	    if ( fabs(x1*y1 +x2*y2) > eps ) {printf("   Matrix U with eigenvectors nor unitary\n");exit(1);}

	    // checking that U * lambdaI *U^T = M

	    double MU11 = lambda1*x1;
	    double MU12 = lambda1*y1;
	    double MU21 = lambda2*x2;
	    double MU22 = lambda2*y2;

            double UMU11 = x1*MU11 + x2*MU21;
	    double UMU12 = x1*MU12 + x2*MU22;
	    double UMU21 = y1*MU11 + y2*MU21;
            double UMU22 = y1*MU12 + y2*MU22;

	    if ( fabs(UMU11-M22) > eps ){printf("eigenvalue decomposition not valid at 11\n");return(0);}
	    if ( fabs(UMU12-M23) > eps ){printf("eigenvalue decomposition not valid at 12\n");return(0);}
	    if ( fabs(UMU21-M23) > eps ){printf("eigenvalue decomposition not valid at 21\n");return(0);}
	    if ( fabs(UMU22-M33) > eps ){printf("eigenvalue decomposition not valid at 22\n");return(0);}

	    // formulating U * sqrt(lambda)
	    
	    double sprime21 = sqrt(lambda1)*x1;
	    double sprime22 = sqrt(lambda2)*x2;
	    double sprime31 = sqrt(lambda1)*y1;
	    double sprime32 = sqrt(lambda2)*y2;
	    double sprime11 = 0.0;
	    double sprime12 = 0.0;


	    double s[4][4];
	    s[1][1] = sprime11 ;
	    s[1][2] = sprime12 ;
	    s[2][1] = sprime21 ;
	    s[2][2] = sprime22 ;
	    s[3][1] = sprime31 ;
	    s[3][2] = sprime32 ;


	    // Center of mass, origin
	    double s_cm[4];

	    for ( int i = 1; i <= 2; ++i ) 
	        {
	        s_cm[i] = 0.0;
		for ( int j = 1; j <= 3 ; ++j ) s_cm[i] += mass[j]*s[j][i];
		s_cm[i] /= (mass[1] + mass[2] + mass[3]);
		}


	    // shift to CMS coordinates
	   
	    for ( int i= 1 ; i <= 3; ++i ) 
		for ( int j = 1; j <= 2 ; ++j ) s[i][j] -= s_cm[j];

	    // Rotate s1 parallel to the x-axis: (s,0.0)
	    double alpha, cosalpha, sinalpha;
            if ( fabs(s[1][1]) < eps ) { cosalpha = 0.0; sinalpha = 1.0;}
	    else
	       {
	       alpha = atan2(s[1][2],s[1][1]);
	       cosalpha = cos(alpha);
	       sinalpha = sin(alpha);
	       }

	    for (int i = 1; i <= 3 ; ++i ) 
	       {
	       double x = s[i][1];
	       double y = s[i][2];
               s[i][1] =  cosalpha*x + sinalpha*y;
	       s[i][2] = -sinalpha*x + cosalpha*y;
	       }

	    if ( s[1][1] < -eps || fabs(s[1][2]) > eps ) {printf("  Rotation of s1 || x-axis did not succeed\n");exit(1);}

	    // checking that the distances have not changed ,,,
	    double dist12 = sqrt( (s[1][1] - s[2][1])*(s[1][1] - s[2][1]) + (s[1][2] - s[2][2])*(s[1][2] - s[2][2]) );
	    double dist13 = sqrt( (s[1][1] - s[3][1])*(s[1][1] - s[3][1]) + (s[1][2] - s[3][2])*(s[1][2] - s[3][2]) );
	    double dist23 = sqrt( (s[2][1] - s[3][1])*(s[2][1] - s[3][1]) + (s[2][2] - s[3][2])*(s[2][2] - s[3][2]) );

	    if ( fabs(dist12 - d12) > eps ) printf(" dist 12 = %lf, d12 = %lf\t",dist12,d12);
	    if ( fabs(dist13 - d13) > eps ) printf(" dist 13 = %lf, d13 = %lf\t",dist13,d13);
	    if ( fabs(dist23 - d23) > eps ) printf(" dist 23 = %lf, d23 = %lf\t",dist23,d23);

	    if ( s[2][2] < 0.0 ) 
	       {
	       printf("Triangle must be reflected wrt x-axis t = %e, d12 = %e, d13 = %e, d23 = %e\n", t,d12,d13,d23);exit(1);}

	    for ( int i = 1 ; i <= 3; ++i ) 
		for ( int j = 1; j <= 2 ; ++j ) shapecoord[i][j] = s[i][j];
 
	    return(0);
	    }

// calculate d\theta/dt, to be integrated
// derivative estimated as 
//      f'(x) ~ ( -f(x+2*h) + 8f(x+h) - 8f(x-h) + f(x-2*h) )/12*h

double dthetadt(double t, double h)
{
	    double dotshape[4][4];
            double shape[4][4];
	    double der[4][4];
            int func(double t, double shapecoord[][4]);
	    int printflag = 1; // 0 = prints, 1 == no printing

            func(t,shape);

            //      f'(x) ~ ( -f(x+2*h) + 8f(x+h) - 8f(x-h) + f(x-2*h) )/12*h
            for ( int i = 1; i <= 3; ++i ) for (int k = 1; k <= 2 ; ++k ) der[i][k] = 0.0;
            func(t+2*h,dotshape);
            for ( int i = 1; i <= 3; ++i ) for (int k = 1; k <= 2 ; ++k ) der[i][k] += -dotshape[i][k];
            func(t+h,dotshape);
            for ( int i = 1; i <= 3; ++i ) for (int k = 1; k <= 2 ; ++k ) der[i][k] += 8.0*dotshape[i][k];
            func(t-h,dotshape);
            for ( int i = 1; i <= 3; ++i ) for (int k = 1; k <= 2 ; ++k ) der[i][k] += -8.0*dotshape[i][k];
            func(t-2*h,dotshape);
            for ( int i = 1; i <= 3; ++i ) for (int k = 1; k <= 2 ; ++k ) der[i][k] += dotshape[i][k];
            for ( int i = 1; i <= 3; ++i ) for (int k = 1; k <= 2 ; ++k ) der[i][k] /= 12.0*h;

	    double  sdotabsval  = sqrt( der[1][1]*der[1][1] + der[1][2]*der[1][2]) +
               			  sqrt( der[2][1]*der[2][1] + der[2][2]*der[2][2]) +
			          sqrt( der[3][1]*der[3][1] + der[3][2]*der[3][2] );

	    if ( minsigmaabssdot >  sdotabsval )
               {
               minsigmaabssdot = sdotabsval;
               minsigmat = t;
               }
            
	    inertia =   mass[1]*(shape[1][1]*shape[1][1] + shape[1][2]*shape[1][2]) 
	              + mass[2]*(shape[2][1]*shape[2][1] + shape[2][2]*shape[2][2]) 
	              + mass[3]*(shape[3][1]*shape[3][1] + shape[3][2]*shape[3][2]);

	
            double crossprod    = + mass[1]*(der[1][1]*shape[1][2] - shape[1][1]*der[1][2])
	                          + mass[2]*(der[2][1]*shape[2][2] - shape[2][1]*der[2][2])
                                  + mass[3]*(der[3][1]*shape[3][2] - shape[3][1]*der[3][2]);
				 

            double innerproduct = mass[1]*(shape[1][1]*shape[1][1] + shape[1][2]*shape[1][2])
                                + mass[2]*(shape[2][1]*shape[2][1] + shape[2][2]*shape[2][2])
                                + mass[3]*(shape[3][1]*shape[3][1] + shape[3][2]*shape[3][2]);

	    sdotabsval  = sqrt( der[1][1]*der[1][1] + der[1][2]*der[1][2]) +
                          sqrt( der[2][1]*der[2][1] + der[2][2]*der[2][2]) +
	                  sqrt( der[3][1]*der[3][1] + der[3][2]*der[3][2] );

            if ( minsigmaabssdot >  sdotabsval ) 
	       {
	       minsigmaabssdot = sdotabsval;
	       minsigmat = t;
	       }

	    return( crossprod/innerproduct);

}


// Polynomial interpolation, from Press, Teukolsky, Vetterling and Flannery: Numerical Recipes in C++ (2nd ed.), p.112
void polint( double xa[], double ya[], int n, double x, double *y, double *dy)
{
        int i,m,ns=1;
        double den,dif,dift,ho,hp,w;
        double c[128],d[128];

        dif=fabs(x-xa[1]);
        for (i=1;i<=n;i++) {
                if ( (dift=fabs(x-xa[i])) < dif) {
                        ns=i;
                        dif=dift;
                }
                c[i]=ya[i];
                d[i]=ya[i];
        }
        *y=ya[ns--];
        for (m=1;m<n;m++) {
                for (i=1;i<=n-m;i++) {
                        ho=xa[i]-x;
                        hp=xa[i+m]-x;
                        w=c[i+1]-d[i];
                        if ( (den=ho-hp) == 0.0) {printf("Error in routine polint"); exit(1);}
                        den=w/den;
                        d[i]=hp*den;
                        c[i]=ho*den;
                }
                *y += (*dy=(2*ns < (n-m) ? c[ns+1] : d[ns--]));
        }
}

double subsum[128];

// Extended trapezoidal integration, from Press, Teukolsky, Vetterling and Flannery: Numerical Recipes in C++ (2nd ed.), p.141
// supplying omp id as argument id this routine is thread safe up to thread # 127 by using the static array subsum[]
double trapzd(double a, double b, int n, double h, double derh, int id)
{
        double x,tnm,sum,del;
        int it,j;

        if (n == 1) {
                return (subsum[id]=0.5*(b-a)*(dthetadt(a,h*derh)+dthetadt(b,h*derh)));
        } else {
                for (it=1,j=1;j<n-1;j++) it <<= 1;
                tnm=it;
                del=(b-a)/tnm;
                x=a+0.5*del;
                for (sum=0.0,j=1;j<=it;j++,x+=del) sum += dthetadt(x,del/10.0);
                subsum[id]=0.5*(subsum[id]+(b-a)*sum/tnm);
                return subsum[id];
        }
}


#define EPS 1.0e-10
#define JMAX 14
#define JMAXP (JMAX+1)
#define K 5

// Romberg integration on an open interval, from Press, Teukolsky, Vetterling and Flannery: Numerical Recipes in C++ (2nd ed.), p.148
// The stopping crieterion has been imporved to cover integrals with values below 10e-10.
double qromo(double (*func)(double,double), double a, double b,
	double (*choose)(double(*)(double,double), double, double, int, double, int), double delta, int id)
{
	void polint(double xa[], double ya[], int n, double x, double *y, double *dy);
	double ss,dss,h[JMAXP+1],s[JMAXP];

	h[1]=1.0;
	for (int j=1;j<=JMAX;j++) {
		s[j]=(*choose)(func,a,b,j,delta,id);
		if (j >= K) {
			polint(&h[j-K],&s[j-K],K,0.0,&ss,&dss);
			if (fabs(dss) <= EPS*fabs(ss) || fabs(ss) < 1.0e-10 || fabs(dss) < 1.0e-10 ) return ss;
		}
		h[j+1]=h[j]/9.0;
	}
	printf("Too many steps in routing qromo, a = %lf b = %lf ss = %.15lf, dss = %.15lf\n",a,b,ss,dss);
	exit(-1);
	return 0.0;
}
#undef EPS
#undef JMAX
#undef JMAXP
#undef K


// the integrand, the time derivative of theta 
double dthetadtsvalues(double t, double h, double scoord[][4], double sder[][4])
{
	    double dotshape[4][4];
            double shape[4][4];
	    double der[4][4];
            int func(double t, double shapecoord[][4]);
	    int printflag = 1; // 0 = prints, 1 == no printing

            func(t,shape);

            //      f'(x) ~ ( -f(x+2*h) + 8f(x+h) - 8f(x-h) + f(x-2*h) )/12*h
            for ( int i = 1; i <= 3; ++i ) for (int k = 1; k <= 2 ; ++k ) der[i][k] = 0.0;
            func(t+2*h,dotshape);
            for ( int i = 1; i <= 3; ++i ) for (int k = 1; k <= 2 ; ++k ) der[i][k] += -dotshape[i][k];
            func(t+h,dotshape);
            for ( int i = 1; i <= 3; ++i ) for (int k = 1; k <= 2 ; ++k ) der[i][k] += 8.0*dotshape[i][k];
            func(t-h,dotshape);
            for ( int i = 1; i <= 3; ++i ) for (int k = 1; k <= 2 ; ++k ) der[i][k] += -8.0*dotshape[i][k];
            func(t-2*h,dotshape);
            for ( int i = 1; i <= 3; ++i ) for (int k = 1; k <= 2 ; ++k ) der[i][k] += dotshape[i][k];
            for ( int i = 1; i <= 3; ++i ) for (int k = 1; k <= 2 ; ++k ) der[i][k] /= 12.0*h;

	    double  sdotabsval  = sqrt( der[1][1]*der[1][1] + der[1][2]*der[1][2]) +
               			  sqrt( der[2][1]*der[2][1] + der[2][2]*der[2][2]) +
			          sqrt( der[3][1]*der[3][1] + der[3][2]*der[3][2] );

	    if ( minsigmaabssdot >  sdotabsval )
               {
               minsigmaabssdot = sdotabsval;
               minsigmat = t;
               }
            
	    inertia =   mass[1]*(shape[1][1]*shape[1][1] + shape[1][2]*shape[1][2]) 
	              + mass[2]*(shape[2][1]*shape[2][1] + shape[2][2]*shape[2][2]) 
	              + mass[3]*(shape[3][1]*shape[3][1] + shape[3][2]*shape[3][2]);

	
            double crossprod    = + mass[1]*(der[1][1]*shape[1][2] - shape[1][1]*der[1][2])
	                          + mass[2]*(der[2][1]*shape[2][2] - shape[2][1]*der[2][2])
                                  + mass[3]*(der[3][1]*shape[3][2] - shape[3][1]*der[3][2]);
				 

            double innerproduct = mass[1]*(shape[1][1]*shape[1][1] + shape[1][2]*shape[1][2])
                                + mass[2]*(shape[2][1]*shape[2][1] + shape[2][2]*shape[2][2])
                                + mass[3]*(shape[3][1]*shape[3][1] + shape[3][2]*shape[3][2]);

	    sdotabsval  = sqrt( der[1][1]*der[1][1] + der[1][2]*der[1][2]) +
                          sqrt( der[2][1]*der[2][1] + der[2][2]*der[2][2]) +
	                  sqrt( der[3][1]*der[3][1] + der[3][2]*der[3][2] );

	    
            if ( minsigmaabssdot >  sdotabsval ) 
	       {
	       minsigmaabssdot = sdotabsval;
	       minsigmat = t;
	       }
	    
            for ( int i = 1; i <= 3; ++i ) 
		    for (int k = 1; k <= 2 ; ++k ) 
			    scoord[i][k] = shape[i][k];

            for ( int i = 1; i <= 3; ++i ) 
		    for (int k = 1; k <= 2 ; ++k ) 
			    sder[i][k]   = der[i][k];

	    return( crossprod/innerproduct);

}

// Extended midpoint rule, from Press, Teukolsky, Vetterling and Flannery: Numerical Recipes in C++ (2nd ed.), p.147
double midpnt(double (*func)(double,double), double a, double b, int n, double delta, int id)
{
	double x,tnm,sum,del,ddel,inert;
	static double s[128];
	int it,j;

	if (n == 1) {
		return (s[id]=(b-a)*func(0.5*(a+b),delta));
	} else {
		for(it=1,j=1;j<n-1;j++) it *= 3;
		tnm=it;
		del=(b-a)/(3.0*tnm);
		ddel=del+del;
		x=a+0.5*del;
		sum=0.0;
		for (j=1;j<=it;j++) {
			sum += func(x,delta);
			x += ddel;
			sum += func(x,delta);
			x += del;
		}
		s[id]=(s[id]+(b-a)*sum/tnm)/3.0;
		return s[id];
	}
}
