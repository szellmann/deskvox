// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

#pragma once


#include "vvcompiler.h"


/*! Declare classes or functions deprecated
 */
#if VV_CXX_GCC && !VV_CXX_INTEL
# define VV_DECL_DEPRECATED __attribute__((__deprecated__))
#elif VV_CXX_MSVC && (VV_CXX_MSVC >= 1300)
# define VV_DECL_DEPRECATED __declspec(deprecated)
#else
# define VV_DECL_DEPRECATED
#endif


/*! Place in private section of class to disallow copying and assignment
 */
#define VV_NOT_COPYABLE(T)                                          \
  T(T const& rhs);                                                  \
  T& operator=(T const& rhs);



/*! Verbose way to say that a parameter is not used intentionally
 */
#define VV_UNUSED(x) ((void)(x))

