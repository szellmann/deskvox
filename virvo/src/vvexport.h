// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, schulze@cs.brown.edu
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

#ifndef VV_EXPORT_H
#define VV_EXPORT_H

// ---------------------------------------------------------------------- //

// defines for windows dll export
#ifndef VIRVOEXPORT
#if defined (_WIN32) && !defined(NODLL)
#if defined (VIRVO_EXPORT)
#define VIRVOEXPORT __declspec(dllexport)
#elif defined (VIRVO_IMPORT)
#define VIRVOEXPORT __declspec(dllimport)
#else
#define VIRVOEXPORT
#endif
#else
#define VIRVOEXPORT
#endif
#endif
#endif                                            /* VV_EXPORT_H */
