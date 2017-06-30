dictionary = {
    # Kernel 23 -- 2-D implicit hydrodynamics fragment
    '2d_hydro': {
        'e': 'z + (0.175 * (a*b + c*d + e*f + g*h + i + j))',
        'v': {'a':[0,1],'b':[0,1],'c':[0,1],'d':[0,1],'e':[0,1],'f':[0,1],'g':[0,1],'h':[0,1],'i':[0,1],'j':[-1,0],'z':[0,1],}
    },


#     /*
#      *******************************************************************
#      *   Kernel 7 -- equation of state fragment
#      *******************************************************************
#      *    DO 7 L= 1,Loop
#      *    DO 7 k= 1,n
#      *      X(k)=     U(k  ) + R*( Z(k  ) + R*Y(k  )) +
#      *   .        T*( U(k+3) + R*( U(k+2) + R*U(k+1)) +
#      *   .        T*( U(k+6) + R*( U(k+5) + R*U(k+4))))
#      *  7 CONTINUE
#      */

#     for ( l=1 ; l<=loop ; l++ ) {
# #pragma nohazard
#         for ( k=0 ; k<n ; k++ ) {
#             x[k] = u[k] + r*( z[k] + r*y[k] ) +
#                    t*( u[k+3] + r*( u[k+2] + r*u[k+1] ) +
#                       t*( u[k+6] + r*( u[k+5] + r*u[k+4] ) ) );
#         }
#     }

	'state_frag': {
        'max_transformation_depth': 3,
        'e': '''u + r*( z + r*y ) +
           t*( a + r*( b + r*c) +
              t*( d + r*( e + r*f ) ) )''',
        'v': {'a':[0,1],'b':[0,1],'c':[0,1],'d':[0,1],'e':[0,1],'f':[0,1],'r':[0,1],'t':[0,1],'u':[0,1],'y':[0,1],'z':[0,1]}
    },


# Kernel 16 -- Monte Carlo search loop
# tmp=(d[j5-1]-(d[j5-2]*(t-d[j5-3])*(t-d[j5-3])+(s-d[j5-4])*
#                               (s-d[j5-4])+(r-d[j5-5])*(r-d[j5-5])));


# Kernel 18 - 2-D explicit hydrodynamics fragment
# zu[k][j] += s*( za[k][j]   *( zz[k][j] - zz[k][j+1] ) -
#                                 za[k][j-1] *( zz[k][j] - zz[k][j-1] ) -
#                                 zb[k][j]   *( zz[k][j] - zz[k-1][j] ) +
#                                 zb[k+1][j] *( zz[k][j] - zz[k+1][j] ) );

}