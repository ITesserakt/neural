use crate::differentiation::{Record, WengertList};
use num_traits::real::Real;
use num_traits::{FromPrimitive, Num, NumCast, One, ToPrimitive, Zero};
use std::cmp::Ordering;
use std::fmt::{Formatter, LowerExp};
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

/**
 * A record is displayed by showing its number component.
 */
impl<'a, T: std::fmt::Display + Copy> std::fmt::Display for Record<'a, T> {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{}", self.number)
    }
}

impl<'a, T: LowerExp + Copy> LowerExp for Record<'a, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.number.fmt(f)
    }
}

impl<T: Copy> Copy for Record<'_, T> {}

/**
 * Compares two record's referenced WengertLists.
 *
 * If either Record is missing a reference to a WengertList then
 * this is trivially 'true', in so far as we will use the WengertList of
 * the other one.
 *
 * If both records have a WengertList, then checks that the lists are
 * the same.
 */
pub(crate) fn same_list<T: Copy>(a: &Record<T>, b: &Record<T>) -> bool {
    match (a.history, b.history) {
        (None, None) => true,
        (Some(_), None) => true,
        (None, Some(_)) => true,
        (Some(list_a), Some(list_b)) => same_lists(list_a, list_b),
    }
}

/// Compares two WengertList references directly.
pub(crate) fn same_lists<T: Copy>(list_a: &WengertList<T>, list_b: &WengertList<T>) -> bool {
    std::ptr::eq(list_a, list_b)
}

impl<N: FromPrimitive + Copy> FromPrimitive for Record<'_, N> {
    fn from_i64(n: i64) -> Option<Self> {
        Some(Record::constant(N::from_i64(n)?))
    }

    fn from_u64(n: u64) -> Option<Self> {
        Some(Record::constant(N::from_u64(n)?))
    }
}

impl<T: PartialEq + Copy> PartialEq for Record<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        self.number == other.number
    }
}

impl<T: PartialOrd + Copy> PartialOrd for Record<'_, T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.number.partial_cmp(&other.number)
    }
}

impl<T: ToPrimitive + Copy> ToPrimitive for Record<'_, T> {
    fn to_i64(&self) -> Option<i64> {
        Some(self.number.to_i64()?)
    }

    fn to_u64(&self) -> Option<u64> {
        Some(self.number.to_u64()?)
    }
}

impl<N: NumCast + Copy> NumCast for Record<'_, N> {
    fn from<T: ToPrimitive>(n: T) -> Option<Self> {
        Some(Record::constant(N::from(n)?))
    }
}

impl<T: Copy> Add<Self> for Record<'_, T>
where
    T: Add<Output = T> + Zero + One + Clone,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        debug_assert!(same_list(&self, &rhs));
        self.binary(rhs, |x, y| x + y, |_, _| T::one(), |_, _| T::one())
    }
}

impl<T: Copy> Zero for Record<'_, T>
where
    T: Zero + One + Clone,
{
    fn zero() -> Self {
        Record::constant(T::zero())
    }

    fn is_zero(&self) -> bool {
        self.number.is_zero()
    }
}

impl<T: Copy> One for Record<'_, T>
where
    T: One + Zero + Clone,
{
    fn one() -> Self {
        Record::constant(T::one())
    }
}

impl<T: Copy> Mul<Self> for Record<'_, T>
where
    T: Mul<Output = T> + Zero + One + Clone,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        debug_assert!(same_list(&self, &rhs));
        self.binary(rhs, |x, y| x * y, |_, y| y, |x, _| x)
    }
}

impl<T: Copy> Sub<Self> for Record<'_, T>
where
    T: Sub<Output = T> + Zero + Clone + One + Neg<Output = T>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        debug_assert!(same_list(&self, &rhs));
        self.binary(rhs, |x, y| x - y, |_, _| T::one(), |_, _| -T::one())
    }
}

impl<T: Copy> Div<Self> for Record<'_, T>
where
    T: Div<Output = T> + Zero + Clone + One + Neg<Output = T>,
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        debug_assert!(same_list(&self, &rhs));
        self.binary(
            rhs,
            |x, y| x / y,
            |_, y| T::one() / y,
            |x, y| -x / (y.clone() * y),
        )
    }
}

impl<T: Copy> Rem<Self> for Record<'_, T>
where
    T: Rem<Output = T> + Zero + Clone + One + Real,
{
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        debug_assert!(same_list(&self, &rhs));
        self.binary(rhs, |x, y| x % y, |_, _| T::one(), |x, y| -(x / y).round())
    }
}

impl<T: Copy> Num for Record<'_, T>
where
    T: PartialEq + Zero + Clone + Real + One + Num,
{
    type FromStrRadixErr = T::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Ok(Record::constant(T::from_str_radix(str, radix)?))
    }
}

impl<T: Copy> Neg for Record<'_, T>
where
    T: Neg<Output = T> + Zero + Clone + One,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.unary(|x| -x, |_| -T::one())
    }
}

#[cfg(test)]
mod tests {
    use crate::differentiation::WengertList;
    use ndarray_linalg::aclose;
    use num_traits::real::Real;
    use num_traits::NumCast;
    use std::ops::{Add, Mul, Sub};

    fn k<T: NumCast + Add<Output = T> + Mul<Output = T> + Copy + Sub<Output = T>>(x: T, y: T) -> T {
        let three = T::from(3).unwrap();

        x + three * y - x * x * y
    }

    #[test]
    fn test_gradient() {
        let tape = WengertList::new();

        let x = tape.variable(1.0);
        let y = tape.variable(2.0);
        let z = k(x, y);

        aclose(z.number, k(1.0, 2.0), 1e-15);
        let ds = z.derivatives();
        aclose(ds[&x], -3.0, 1e-15);
        aclose(ds[&y], 2.0, 1e-15);
    }

    fn f<T: Real>(x: T, y: T) -> T {
        x.sin() * y.cos()
    }

    fn g<T: Real>(x: T) -> T {
        (x + T::from(2).unwrap()) * x.ln()
    }

    #[test]
    fn test_simple_derivative() {
        let tape = WengertList::new();
        let x = tape.variable(0.4);

        let y = x.unary(g, |x| x.ln() + (x + 2.0) / x);
        aclose(y.number, -2.199097756497972, 1e-15);

        let ds = y.derivatives();
        aclose(ds[&x], 5.083709268125845, 1e-15);
    }

    #[test]
    fn test_compound_derivative() {
        let tape = WengertList::new();
        let x = tape.variable(0.4);
        let y = tape.variable(0.7);

        let z = x.binary(y, f, |x, y| x.cos() * y.cos(), |x, y| -y.sin() * x.sin());
        aclose(z.number, 0.2978435767000479, 1e-15);

        let ds = z.derivatives();
        aclose(ds[&x], 0.7044663052755917, 1e-15);
        aclose(ds[&y], -0.2508701838500143, 1e-15);
    }
}
